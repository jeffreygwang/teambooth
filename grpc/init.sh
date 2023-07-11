git clone https://github.com/ShivamShrirao/diffusers.git
cd diffusers/examples/dreambooth/
python3 -m venv env
source env/bin/activate
pip install git+https://github.com/ShivamShrirao/diffusers.git
pip install -U -r requirements.txt
sudo yum -y install python3-devel
sudo yum -y install bsdiff
pip install deepspeed
pip install bsdiff4
pip install boto3
pip install omegaconf
pip uninstall urllib3
pip install urllib3==1.26.15
accelerate config
echo "Now cd diffusers/examples/dreambooth/ etc."
