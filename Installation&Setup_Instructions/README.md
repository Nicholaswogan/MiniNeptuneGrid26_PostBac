
# First time setup

```sh
conda env create -f environment.yaml
conda activate subneptune

# Install picaso
wget https://github.com/Nicholaswogan/picaso/archive/837b04c3432133189697f5ed1a28a3c62f364e61.zip
unzip 837b04c3432133189697f5ed1a28a3c62f364e61.zip
cd picaso-837b04c3432133189697f5ed1a28a3c62f364e61
python -m pip install . -v
# Get reference
cd ../
mkdir picasofiles
cp -r picaso-837b04c3432133189697f5ed1a28a3c62f364e61/reference picasofiles/reference
rm -rf picaso-837b04c3432133189697f5ed1a28a3c62f364e61
rm 837b04c3432133189697f5ed1a28a3c62f364e61.zip

# Get the star stuff
wget http://ssb.stsci.edu/trds/tarfiles/synphot3.tar.gz
tar -xvzf synphot3.tar.gz
mv grp picasofiles/
rm synphot3.tar.gz

# setup
export picaso_refdata=$(pwd)"/picasofiles/reference/" 
export PYSYN_CDBS=$(pwd)"/picasofiles/grp/redcat/trds"

```

# Every other time

```sh
conda activate subneptune
export picaso_refdata=$(pwd)"/picasofiles/reference/"
export PYSYN_CDBS=$(pwd)"/picasofiles/grp/redcat/trds"
```
