mkdir -p data
mkdir -p data/MSLR-WEB30K
for i in $(seq 10 1 21)
do
    ln -s /Volumes/Data/chen/amaris/ethz/spr2019/fastcode/project/basic/master/data/MSLR-WEB30K/$i.train data/MSLR-WEB30K/$i.train
    ln -s /Volumes/Data/chen/amaris/ethz/spr2019/fastcode/project/basic/master/data/MSLR-WEB30K/$i.train.query data/MSLR-WEB30K/$i.train.query
done

ln -s /Volumes/Data/chen/amaris/ethz/spr2019/fastcode/project/basic/master/data/MSLR-WEB30K/17.vali data/MSLR-WEB30K/17.vali
ln -s /Volumes/Data/chen/amaris/ethz/spr2019/fastcode/project/basic/master/data/MSLR-WEB30K/17.vali.query data/MSLR-WEB30K/17.vali.query
