#!/bin/bash
cd /projects/luckin_purchase_jupyter || exit
name=$(hostname)
host="luckymlpurchase01-prod-py-vianetm5"
if ((name == host))
then
 git checkout --force master
 git pull origin master
 echo 'RUN master'
 export PATH=/data1/user/anaconda3/envs/py3_8_5/bin:$PATH
else
 git checkout --force develop
 git pull origin develop
 echo 'RUN develop'
fi