cd dataset/image_sets
find ../data_syn -name "*-color.png" | sed 's/\(-color.png\)$//' > synthetic.txt
cat train.txt synthetic.txt > trainsyn.txt