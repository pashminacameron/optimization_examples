# Modified version of installation script from http://dirk.eddelbuettel.com/blog/2018/04/15/
# Run as sudo ./installmkl.sh

echo "Setting up repos for MKL"
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
apt-get update
echo "Installing MKL"
apt-get install intel-mkl-64bit-2018.3-051
echo "Adding MKL to LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/lib/intel64:/opt/intel/mkl/lib/intel64
echo "MKL setup done"


