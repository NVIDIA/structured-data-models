echo "f1-dnf ========================="
python rel_bench.py --dataset=rel-f1 --task=driver-dnf
echo "f1-top3 ========================"
python rel_bench.py --dataset=rel-f1 --task=driver-top3
echo "event-ignore ==================="
python rel_bench.py --dataset=rel-event --task=user-ignore
echo "event-repeat ==================="
python rel_bench.py --dataset=rel-event --task=user-repeat
echo "avito-clicks ==================="
python rel_bench.py --dataset=rel-avito --task=user-clicks
echo "avito-visits ==================="
python rel_bench.py --dataset=rel-avito --task=user-visits
echo "hm-churn ======================="
python rel_bench.py --dataset=rel-hm --task=user-churn
echo "trial-outcome =================="
python rel_bench.py --dataset=rel-trial --task=study-outcome
echo "stack-badge ===================="
python rel_bench.py --dataset=rel-stack --task=user-badge
echo "stack-engagement ==============="
python rel_bench.py --dataset=rel-stack --task=user-engagement
# echo "amazon-item ===================="
# python rel_bench.py --dataset=rel-amazon --task=item-churn
# echo "amazon-user ===================="
# python rel_bench.py --dataset=rel-amazon --task=user-churn
