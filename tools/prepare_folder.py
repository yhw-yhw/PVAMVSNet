import os
import argparse
import sys
parser = argparse.ArgumentParser(
    description='prepare folder')
parser.add_argument('--result', default='/data1/wzz/down4/', help='work space')

args = parser.parse_args()

result=args.result

sub_folders=os.listdir(result)
down0=os.path.join(result,'0')
down1=os.path.join(result,'1')
down2=os.path.join(result,'2')
if not os.path.exists(down0):
    os.makedirs(down0)
if not os.path.exists(down1):
    os.makedirs(down1)
if not os.path.exists(down2):
    os.makedirs(down2)
for sub_folder in sub_folders:
    input=os.path.join(result,sub_folder)
    for i in range(3):
        input_s=os.path.join(input,'depth_est_%d'%i)
        output_s=os.path.join(result,'%d'%i)
        output_s=os.path.join(output_s,sub_folder)
        if not os.path.exists(output_s):
            os.makedirs(output_s)
        output_s=os.path.join(output_s,'depth_est')
        os.system('cp -r '+input_s+' '+output_s)

        input_s = os.path.join(input, 'confidence_%d' % i)
        output_s = os.path.join(result, '%d' % i)
        output_s = os.path.join(output_s, sub_folder)
        if not os.path.exists(output_s):
            os.makedirs(output_s)
        output_s = os.path.join(output_s, 'confidence')
        os.system('cp -r ' + input_s + ' ' + output_s)
