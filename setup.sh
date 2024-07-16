conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia

mkdir lib
cd lib

git clone --recursive https://github.com/vye16/slahmr.git
cd slahmr
git checkout e54e77a121bf1078afafaf79d018445e6aca44a5

cd third-party
git -b dev_slahmr clone https://github.com/brjathu/PHALP.git
cd ..
pip install -v -e 'third-party/PHALP[all]'

pip install -v -e third-party/ViTPose

cd third-party/DROID-SLAM
python setup.py install
cd ../../

pip install -e .

sh ./download_models.sh
cd ../..

pip install -r requirements.txt

mv lib/slahmr/_DATA/body_models lib/slahmr