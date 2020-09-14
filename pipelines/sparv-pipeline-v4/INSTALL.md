
 #### Install prerequisites

 ```bash
sudo apt-get install git-lfs
pyenv shell 3.8.5
python -m pip install pipx --user
python -m pipx ensurepath
 ```

#### Install Sparv v4

```bash
pyenv shell 3.8.5
pipx install https://github.com/spraakbanken/sparv-pipeline/archive/latest.tar.gz
```
or

```bash
pyenv shell 3.8.5
git clone https://github.com/spraakbanken/sparv-pipeline.git
cd sparv-pipeline
git checkout v4
pipx install .
```

#### Setup Sparv Data Folder

```bash
$ sparv setup
Sparv needs a place to store its configuration files, language models and other data. Enter the path to the directory you want to use. Leave empty to continue using '/home/xyz/.local/share/sparv'.

# Use SPARV_DIR environment variable to override configured data dir:
export SPARV_DIR=/home/xyz/.local/share/sparv

```

#### Install 3rd-party Software

Maltparser

```bash
cd $SPARV_DIR/bin

wget -qO- http://maltparser.org/dist/maltparser-1.7.2.tar.gz | tar xvz
```

Hunpos

```bash
wget -qO- https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/hunpos/hunpos-1.0-linux.tgz | tar xvz
```

or compile

```bash
sudo apt-get update
sudo apt-get install -yqq --no-install-recommends cmake ocaml-nox

cd $SPARV_DIR/bin

git clone https://github.com/mivoq/hunpos.git
cd hunpos/
mkdir -p build
cd build/
cmake .. -DCMAKE_INSTALL_PREFIX=install
make && make install
cd ..
cp hunpos/build/hunpos-tag hunpos/build/hunpos-train .
rm -rf ./hunpos/
```

Sparv-wsd

```bash
cd $SPARV_DIR/bin
mkdir -p wsd
cd wsd
wget https://github.com/spraakbanken/sparv-wsd/raw/master/bin/saldowsd.jar
```

HSFT-SweNER

```bash

sudo apt-get install m4

if [-f /usr/bin/python2 ]; then
    sudo apt-get install python2.7-minimal
    sudo ln -s /usr/bin/python2.7 /usr/bin/python2
fi

cd $SPARV_DIR/bin

wget -qO- http://www.ling.helsinki.fi/users/janiemi/finclarin/ner/hfst-swener-0.9.3.tgz | tar xvz

cd hfst-swener-0.9.3/scripts
sed -i 's:#! \/usr/bin/env python:#! /usr/bin/env python2:g' *.py
cd ..

./configure
make

sudo make install

# make clean
```

Corpus Workbench

```bash

cd $SPARV_DIR/bin

svn co http://svn.code.sf.net/p/cwb/code/cwb/trunk cwb

cd cwb
less INSTALL

sudo apt-get install bison flex libpcre3-dev libglib2.0 glib-2.0 libreadline libncurses5-dev libncursesw5-dev
sudo ./install-scripts/install-linux

```

CWB data folder

```bash
mkdir -p ~/cwb/data
export CWB_DATADIR=~/cwb/data;
export CORPUS_REGISTRY=~/cwb/registry

```

#### Build Swedish Models

```bash
sparv build-models --language swe
```
