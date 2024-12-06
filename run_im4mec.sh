#run the im4mec training (+ crossval)


for fold in 0  #1 2 3 4
do
python train.py \
--manifest "train.csv" \
--data_dir "/mnt/bulk-curie/arianna/HECTOR/feature_CONCH/" \
--fold ${fold}
done


#--input_feature_size 1024 \