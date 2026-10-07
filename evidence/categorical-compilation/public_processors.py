"""Public processor compilation probe; CPU Inductor, no graph suppression."""
import copy
import json
import traceback

import torch
from torch._dynamo.backends.registry import lookup_backend

from sdm import CategoricalTensor, NaT, StringTensor, TableTensor
import sdm.processing as sp


def categorical(string=False, query=False, rows=4):
    values = ['blue', 'new', 'red'] if query else ['red', 'blue', 'unused']
    categories = StringTensor.from_list(values) if string else torch.tensor([20, 40, 10] if query else [10, 20, 30])
    codes = (torch.arange(rows) % 3).view(-1, 1).to(torch.int32)
    codes[-1] = -1
    return TableTensor(categorical=CategoricalTensor(codes, categories=[categories]))


def calendar(rows=4):
    dates = torch.tensor([0, 1709164800000000, NaT, -86400000000], dtype=torch.long)
    return TableTensor(datetime=dates[torch.arange(rows) % 4].view(-1, 1))


def assert_same(actual, expected):
    assert actual.columns == expected.columns
    torch.testing.assert_close(actual.numerical, expected.numerical, equal_nan=True)
    torch.testing.assert_close(actual.datetime, expected.datetime)
    torch.testing.assert_close(actual.categorical.code, expected.categorical.code)
    for a, b in zip(actual.categorical.categories, expected.categorical.categories, strict=True):
        assert a.tolist() == b.tolist()


def run():
    cases = {
        'align_numeric': (sp.AlignCategories, lambda rows, query: categorical(query=query, rows=rows)),
        'align_string': (sp.AlignCategories, lambda rows, query: categorical(string=True, query=query, rows=rows)),
        'counts_numeric': (sp.AddCategoryCounts, lambda rows, query: categorical(rows=rows)),
        'shuffle_numeric': (sp.ShuffleCategories, lambda rows, query: categorical(rows=rows)),
        'numerical': (sp.ToNumerical, lambda rows, query: categorical(rows=rows)),
        'calendar': (lambda: sp.AddCalendarFields(['minute','hour','weekday','month','day_of_month'], encoding='cyclic'), lambda rows, query: calendar(rows)),
    }
    for name, (constructor, make_table) in cases.items():
        for method in ('fit_transform', 'transform'):
            for fullgraph in (False, True):
                torch._dynamo.reset()
                graphs = []
                def backend(graph, inputs):
                    graphs.append(len(list(graph.graph.nodes)))
                    return lookup_backend('inductor')(graph, inputs)
                proc = constructor()
                if method == 'transform':
                    proc.fit(make_table(4, False))
                compiled = torch.compile(getattr(proc, method), backend=backend, fullgraph=fullgraph, dynamic=True)
                result = {'version': torch.__version__, 'case': name, 'method': method, 'fullgraph': fullgraph, 'graphs': graphs}
                try:
                    for rows in (4, 6, 3):
                        table = make_table(rows, method == 'transform')
                        oracle = copy.deepcopy(proc)
                        actual = compiled(table)
                        # Shuffle fits new random permutations. Verify the same fitted state
                        # produces exactly the same output outside compilation.
                        expected = proc.transform(table) if name == 'shuffle_numeric' else getattr(oracle, method)(table)
                        assert_same(actual, expected)
                    result['status'] = 'pass'
                except Exception as error:
                    result.update(status='fail', error=type(error).__name__, message=str(error), traceback=traceback.format_exc())
                print(json.dumps(result), flush=True)

if __name__ == '__main__':
    run()
