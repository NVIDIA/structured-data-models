# Vorschlag: Standard für öffentliche Docstrings

Status: Zur Diskussion

Diese Spezifikation schlägt den Docstring-Standard für die öffentliche API von `sdm` vor. Sie bleibt während der Diskussion bewusst von `SKILL.md` getrennt. Nach der Freigabe sollen die akzeptierten Regeln in den Skill übernommen und durch Repository-Checks abgesichert werden.

## Kernaussage

Der Skill erklärt Menschen und Agenten, wie gute Docstrings geschrieben werden. Verbindliche CI-Checks stellen sicher, dass das prüfbare Ergebnis auf `main` dem Standard entspricht. Die Nutzung eines Skills allein ist keine ausreichende Garantie, weil sie nicht in jedem menschlichen, Editor- oder Agenten-Workflow zuverlässig beobachtbar ist.

## Priorisierung nach Impact

| Priorität | Bereich                                | Impact             | Warum dieser Bereich den größten Hebel hat                                                                                       |
| --------- | -------------------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| 1         | Verbindliche Durchsetzung in CI        | Sehr hoch          | Verhindert neue Abweichungen in jedem zukünftigen PR und wirkt unabhängig davon, wer den Docstring geschrieben hat.              |
| 2         | Vollständiger öffentlicher API-Vertrag | Sehr hoch          | Verhindert falsche oder unvollständige Dokumentation direkt an der Schnittstelle, die Nutzer tatsächlich aufrufen.               |
| 3         | Tensor- und Processor-Semantik         | Hoch               | Shapes, Dtypes, Devices und Lifecycle sind die wahrscheinlichsten Quellen für trotz formal korrekter Docstrings falsche Nutzung. |
| 4         | Ausführbare Beispiele                  | Hoch               | Erkennt Drift zwischen Dokumentation und Verhalten und zeigt gleichzeitig den kleinsten korrekten Nutzungspfad.                  |
| 5         | Wiederverwendung und Generierung       | Mittel             | Reduziert Wiederholungen bei wachsender API, lohnt sich aber erst bei tatsächlich identischer Semantik.                          |
| 6         | Markup und sprachliche Konsistenz      | Mittel bis niedrig | Verbessert Lesbarkeit und Rendering, verhindert aber seltener fachlich falsche Nutzung als die vorherigen Bereiche.              |

Die Reihenfolge beschreibt den erwarteten Nutzen für das Projekt, nicht die Implementierungsreihenfolge einzelner Zeilen.

## Impact 1: Verbindliche Durchsetzung in CI

### Warum dieser Block zuerst kommt

Diese Regeln haben den höchsten Impact, weil sie einmal implementiert für jede zukünftige Änderung gelten. Sie verhindern, dass der Standard von der freiwilligen Nutzung eines Skills oder von manueller Aufmerksamkeit im Review abhängt.

- **ENF-01 (MUSS):** `AGENTS.md` weist Menschen und Agenten bei Änderungen an der öffentlichen API unter `sdm/` ausdrücklich an, vor der Arbeit `.agents/skills/docstring/SKILL.md` zu lesen.
- **ENF-02 (MUSS):** Ruff läuft mit der bestehenden Google-Docstring-Konvention über alle Python-Dateien in der verpflichtenden Pull-Request-CI.
- **ENF-03 (MUSS):** Ein Google-Style-Contract-Checker prüft ganz `sdm` und gleicht dokumentierte Argumente, Konstruktorparameter, Rückgabewerte, Generatorwerte und unterstützte Exceptions mit dem Code ab. `pydoclint` ist die vorgeschlagene allgemeine Implementierung.
- **ENF-04 (MUSS):** Ein kleiner repository-eigener Checker prüft objektive SDM-Regeln, die allgemeine Linter nicht ausdrücken können. Dazu gehören mindestens die öffentliche `__all__`-Oberfläche und die Dokumentation von Konstruktorparametern im Klassendocstring.
- **ENF-05 (MUSS):** Der Sphinx-HTML-Build behandelt Warnungen als Fehler, damit ungültiges Markup und nicht auflösbare Referenzen einen Merge blockieren.
- **ENF-06 (MUSS):** Sphinx-Doctests laufen in der verpflichtenden Pull-Request-CI, damit ausführbare Beispiele nicht vom tatsächlichen Verhalten abweichen.
- **ENF-07 (MUSS):** Branch Protection verlangt ENF-02 bis ENF-06 vor dem Merge nach `main`. Eine Prüfung erst nach dem Push auf `main` ist nicht ausreichend.
- **ENF-08 (MUSS):** Nicht zuverlässig automatisierbare Regeln bleiben explizite Review-Punkte. Dazu gehören insbesondere überraschendes Verhalten, fachlich korrekte Shapes und die passende Paper-Referenz.
- **ENF-09 (SOLLTE):** Wenn die Laufzeit es erlaubt, prüfen die Checks das gesamte Repository statt nur den Diff. Änderungen an Signaturen oder Exports können Docstrings außerhalb der bearbeiteten Zeilen ungültig machen.

## Impact 2: Vollständiger öffentlicher API-Vertrag

### Warum dieser Block den zweitgrößten Impact hat

Nutzer lesen Docstrings, um eine API ohne Kenntnis ihrer Implementierung korrekt aufzurufen. Fehlende Parameter, falsche Rückgabewerte oder versprochene Validierungen verursachen deshalb unmittelbar falsche Nutzung und erschweren spätere API-Änderungen.

### Umfang und Zuständigkeit

- **DOC-01 (MUSS):** Ein Objekt gehört zur öffentlichen API, wenn es über die dokumentierte `__all__`-Hierarchie exportiert wird. Öffentliche API und generierte API-Referenz SOLLTEN übereinstimmen.
- **DOC-02 (MUSS):** Jede öffentliche Klasse, Funktion und nicht geerbte öffentliche Methode besitzt einen Docstring.
- **DOC-03 (MUSS):** Öffentliche Konstruktorparameter werden im Klassendocstring dokumentiert und nicht in `__init__` dupliziert.
- **DOC-04 (MUSS):** Eine überschreibende Methode dupliziert keinen geerbten Docstring, wenn dessen Vertrag unverändert gilt. In diesem Fall wird gezielt `# noqa: D102` verwendet. Ändert das Override öffentliches Verhalten, MUSS es den geänderten Vertrag dokumentieren.
- **DOC-05 (SOLLTE):** Einzelne Implementierungsmodule erhalten keine Modul-Docstrings. `__init__.py`-Dateien DÜRFEN eine kurze Package-Zusammenfassung enthalten.

### Inhalt und Struktur

- **DOC-06 (MUSS):** Der Docstring beginnt mit einem kurzen, eigenständig verständlichen Satz. Funktionen und Methoden verwenden die imperative Form, zum Beispiel `Return the encoded table.` Klassen beschreiben ihren Zweck direkt und vermeiden Formulierungen wie `This class ...`.
- **DOC-07 (MUSS):** Ein `Args:`-Abschnitt ist vorhanden, wenn der dokumentierte Callable oder Konstruktor öffentliche Parameter besitzt. Er dokumentiert jeden öffentlichen Parameter und keinen Parameter, der nicht in der Signatur vorkommt. `*args` und `**kwargs` werden dokumentiert, wenn sie Teil des öffentlichen Vertrags sind.
- **DOC-08 (MUSS):** `Returns:` wird verwendet, wenn der Rückgabewert nicht vollständig aus der Summary hervorgeht. `Yields:` beschreibt erzeugte Werte. Callables, die ausschließlich `None` zurückgeben, benötigen keinen dieser Abschnitte.
- **DOC-09 (MUSS):** `Raises:` dokumentiert ausschließlich Exceptions, die Teil des unterstützten öffentlichen Vertrags sind. Zufällige Downstream-Fehler und Exceptions, die ausschließlich durch die Verletzung einer bereits dokumentierten Vorbedingung entstehen, werden nicht dokumentiert.
- **DOC-10 (SOLLTE):** `Attributes:` dokumentiert öffentlichen gefitteten oder gelernten Zustand, den Nutzer untersuchen sollen. Private Caches und zufälliger Implementierungszustand werden nicht dokumentiert.
- **DOC-11 (SOLLTE):** `See Also:` wird verwendet, wenn eine eng verwandte öffentliche API bei der Auswahl der richtigen Operation hilft. Jeder Eintrag erklärt kurz die Beziehung; reine Linklisten sind nicht erlaubt.
- **DOC-13 (MUSS):** API-Docstrings bleiben auf den Referenzvertrag beschränkt. Ausführliche Tutorials, Motivation, Vergleiche und End-to-End-Abläufe gehören in die narrative Dokumentation.

### Parameter, Defaults und Typen

- **DOC-14 (MUSS):** Parameterbeschreibungen erklären Semantik und wiederholen keine bereits präzise vorhandenen Type Hints. Akzeptierte Dtype-Familien, Shapes, Einheiten, Bereiche oder Werte werden nur erwähnt, wenn sie für die korrekte Nutzung relevant sind.
- **DOC-15 (MUSS):** Die Bedeutung von `None`, Sentinel-Werten und besonderen Literalen wird erklärt. Eine reine Wiederholung von `default=None` ist nicht ausreichend.
- **DOC-16 (MUSS):** Implizite Begrenzungen, Fallbacks, Transformationen und Side Effects, die Ergebnisse verändern oder Nutzer überraschen können, werden dokumentiert.
- **DOC-17 (MUSS):** Die Dokumentation verspricht keine breitere Eingabeunterstützung als die Implementierung und erzeugt keine neuen Anforderungen an Runtime-Validierungen, die für den öffentlichen Vertrag nicht nötig sind.

## Impact 3: Tensor- und Processor-Semantik

### Warum dieser Block besonders relevant für SDM ist

Allgemeine Docstring-Linter können einen formal vollständigen Docstring erkennen, aber nicht beurteilen, ob ein Processor vor `transform()` gefittet werden muss oder ob ein Tensor Dtype, Device, Shape oder Stype verändert. Genau diese Eigenschaften entscheiden in SDM häufig darüber, ob eine Verarbeitung korrekt ist.

- **DOC-18 (MUSS):** Tensorparameter und -rückgabewerte enthalten ihre relevante Shape in Double-Backtick-Notation, zum Beispiel `[..., S, C]`. Batchdimensionen beginnen mit `...`; alle übrigen Dimensionsbuchstaben werden konsistent ausgeschrieben.
- **DOC-19 (MUSS):** Dtype- und Device-Verhalten wird dokumentiert, wenn eine Operation Daten einschränkt, promotet, konvertiert oder verschiebt. Dtypes werden nicht aufgezählt, wenn die Operation alle unterstützten Dtypes transparent akzeptiert oder erhält.
- **DOC-20 (MUSS):** Stateful Processor dokumentieren, ob `fit()` erforderlich ist, welcher öffentliche Zustand gelernt wird und ob ein erneutes `fit()` diesen Zustand ersetzt, sofern dies nicht vollständig durch den Basisklassenvertrag definiert ist.
- **DOC-21 (MUSS):** Relevantes Nutzerverhalten für Missing- oder Non-finite-Werte, stochastische Operationen und Generatorsteuerung, Spaltenreihenfolge oder -namen, Stypes und Schemaänderungen wird dokumentiert.
- **DOC-22 (MUSS):** Eingabe- und Ausgabebeschreibungen erklären, ob sich Containerstruktur, Dtype, Device, Shape oder semantischer Spaltentyp verändern, wenn dieses Verhalten nicht offensichtlich ist.

## Impact 4: Ausführbare Beispiele

### Warum Beispiele einen hohen, aber nicht den höchsten Impact haben

Ausführbare Beispiele prüfen Verhalten und Dokumentation gemeinsam und helfen Nutzern schneller als zusätzliche Prosa. Sie verursachen jedoch Pflege- und CI-Kosten. Deshalb sollen sie gezielt für neue Top-Level-APIs und nicht für jede triviale Methode verlangt werden.

- **DOC-12 (SOLLTE):** Ein neuer öffentlicher Top-Level-Processor oder eine API mit nicht offensichtlicher Komposition oder Lifecycle erhält ein minimales deterministisches Beispiel.
- **EX-01 (MUSS):** Das Beispiel läuft ohne Netzwerkzugriff und auf CPU, sofern die dokumentierte Funktion nicht ausdrücklich eine andere Umgebung benötigt.
- **EX-02 (MUSS):** Das Beispiel ist deterministisch oder kontrolliert seine Zufälligkeit explizit.
- **EX-03 (MUSS):** Das Beispiel zeigt die öffentliche API und keine privaten Hilfskonstruktionen oder internen Zustände.
- **EX-04 (MUSS):** Das Beispiel wird vom Sphinx-Doctest-Build ausgeführt, sofern es nicht ausdrücklich und begründet als nicht ausführbar markiert ist.
- **EX-05 (SOLLTE):** Das kleinste realistische Beispiel macht den Ein- und Ausgabevertrag verständlich.

## Impact 5: Wiederverwendung und Generierung

### Warum dieser Block erst später relevant wird

Generierung bietet den größten Nutzen bei vielen identischen Beschreibungen, wie in großen Model Zoos. Zu frühe Templates erzeugen dagegen zusätzliche Abstraktion und können Unterschiede zwischen Processors verdecken. Für SDM sollte deshalb zuerst Konsistenz gemessen und erst danach Wiederholung zentralisiert werden.

- **GEN-01 (MUSS):** Beschreibungen öffentlicher Verträge sind reviewter Quelltext. Generierte API-Seiten DÜRFEN Docstrings, Signaturen, Type Hints und Autosummary-Metadaten verwenden; generierte Prosa ist ohne Review nicht maßgeblich.
- **GEN-02 (SOLLTE):** Wiederverwendbare Parameterbeschreibungen oder Templates werden nur für nachweislich identische Semantik mehrerer APIs eingeführt.
- **GEN-03 (MUSS):** Eine generierte Beschreibung besitzt genau eine versionierte Quelle und einen Konsistenzcheck für Signaturen, Defaults und dokumentierte Parameter.
- **GEN-04 (MUSS):** LLMs oder Agenten DÜRFEN Autoren unterstützen. Die erfolgreiche Generierung oder Skill-Nutzung ist jedoch kein Merge-Kriterium; das Ergebnis durchläuft dieselben Reviews und CI-Checks wie manuell geschriebener Text.

## Impact 6: Markup, Referenzen und sprachliche Konsistenz

### Warum dieser Block nach den semantischen Regeln kommt

Konsistentes Markup verbessert Lesbarkeit, Navigation und das gerenderte Ergebnis. Ein sprachlich perfekter Docstring kann trotzdem fachlich falsch sein, weshalb diese Regeln nicht vor den API- und Processor-Verträgen priorisiert werden sollten.

- **DOC-23 (MUSS):** Docstrings mit Mathematik, LaTeX oder Backslashes verwenden `r"""..."""`.
- **DOC-24 (MUSS):** Implementiert eine API eine in einem wissenschaftlichen Paper vorgeschlagene Methode, wird dieses Paper möglichst über eine stabile URL im ersten Satz zitiert.
- **DOC-25 (SOLLTE):** Auflösbare Sphinx-Cross-References werden für öffentliche interne und über Intersphinx verfügbare externe Ziele gegenüber einfachen Literalen bevorzugt.
- **DOC-26 (MUSS):** `.. deprecated::`, `.. versionchanged::` oder `.. versionadded::` werden verwendet, wenn die entsprechende Information zum öffentlichen API-Lifecycle dauerhaft in der Referenz sichtbar sein muss.
- **DOC-27 (MUSS):** Ein gezieltes `# noqa: <code>` wird nur verwendet, wenn die konkrete Regel absichtlich nicht anwendbar ist. Breite oder unbegründete Suppressions sind nicht erlaubt.

## Einführung nach erwartetem Nutzen

1. Diesen Vorschlag reviewen und die offenen Entscheidungen klären.
2. Akzeptierte Regeln in `SKILL.md` übernehmen und die explizite Skill-Routing-Regel in `AGENTS.md` ergänzen.
3. Den vorgeschlagenen Contract-Checker über ganz `sdm` ausführen und bestehende Findings klassifizieren.
4. Bestehende Verstöße an der öffentlichen API korrigieren, soweit dies mit überschaubarem Aufwand möglich ist.
5. Falls die sofortige Bereinigung zu groß ist, vorübergehend eine Baseline einchecken, die bestehende Verstöße zulässt, aber neue oder verschlechterte Verstöße ablehnt.
6. Die Checks in Pre-commit aufnehmen, wenn sie schnell genug sind, und in jedem Fall als verpflichtende Pull-Request-Checks konfigurieren.
7. Eine vorübergehende Baseline schrittweise abbauen.

## Offene Entscheidungen nach Impact

1. **Sehr hoch:** Soll ENF-03 `pydoclint` verwenden oder sollen alle Contract-Checks in einem repository-eigenen Tool implementiert werden?
2. **Sehr hoch:** Kann der aktuelle Stand die neuen Checks sofort erfüllen oder ist vorübergehend eine Baseline nötig?
3. **Sehr hoch:** Soll `__all__` für DOC-01 allein die öffentliche API definieren oder muss ein Objekt zusätzlich in der generierten API-Referenz enthalten sein?
4. **Hoch:** Welche `Raises:`-Checks können aktiviert werden, ohne defensive Runtime-Validierung zu fördern oder nicht unterstützte Eingaben zum öffentlichen Vertrag zu machen?
5. **Hoch:** Soll DOC-12 für jeden neuen öffentlichen Top-Level-Processor gelten oder nur, wenn die Nutzung nach Einschätzung des Reviews nicht offensichtlich ist?
6. **Hoch:** Soll DOC-19 Dtype- und Device-Aussagen für jede Tensoroperation verlangen oder wie vorgeschlagen nur für Einschränkungen und Transformationen?

## Quellen und Auswahlbegründung

- [PEP 257](https://peps.python.org/pep-0257/) definiert die grundlegende Python-Struktur und Summary-Konventionen, auf denen andere Standards aufbauen.
- Der [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#s3.8.1-comments-in-doc-strings) wurde ausgewählt, weil er zur bestehenden SDM-Syntax mit `Args:`, `Returns:`, `Yields:` und `Raises:` passt.
- [numpydoc validation](https://numpydoc.readthedocs.io/en/stable/validation.html) zeigt, wie versionierte und konfigurierbare Docstring-Prüfungen in Pre-commit und Sphinx eingebunden werden.
- Die [pandas-Dokumentationsrichtlinien](https://pandas.pydata.org/docs/dev/development/contributing_documentation.html) wurden wegen ihrer strikten Sphinx-Validierung und ausführbaren Beispiele für eine große öffentliche API ausgewählt.
- Die [scikit-learn-Docstring-Richtlinien](https://scikit-learn.org/stable/developers/contributing.html#guidelines-for-writing-docstrings) sind wegen der SDM ähnlichen Estimator- und Processor-APIs besonders relevant, unter anderem für Shapes, Dtypes, Defaults, gefittete Attribute und Beispiele.
- Die [JAX-Doctest-Richtlinien](https://github.com/jax-ml/jax/blob/main/docs/developer.md#doctests) zeigen ausführbare Dokumentation in einer modernen Tensor- und Accelerator-Library.
- Der [PyTorch-Docstring-Linter](https://github.com/pytorch/pytorch/blob/main/tools/linter/adapters/docstring_linter.py) zeigt eine schrittweise Einführung über eine Grandfather-Baseline im technisch engsten Runtime-Ökosystem von SDM.
- Die [Transformers-Auto-Docstrings](https://github.com/huggingface/transformers/blob/main/docs/source/en/auto_docstring.md) zeigen signatur- und templatebasierte Generierung in einem großen Model- und Processor-Zoo und machen sichtbar, dass diese Lösung vor allem bei vielfach wiederholter Semantik lohnt.
- [`pydoclint`](https://jsh9.github.io/pydoclint/) unterstützt Google-Style-Prüfungen für Signaturen, Argumente, Rückgaben, Generatorwerte, Exceptions, Konstruktoren und Attribute sowie eine schrittweise Einführung über eine Baseline.
