set OUT=./Segmentation_180408
@rem set MODEL=result_180408\model_latest
set MODEL=result_180408\model_epoch-0970
set GPU=0
set VLIST=ng_pair_list_180408.txt

python inferLunchpack.py %VLIST% -R=%OUT% -m=%MODEL% -g=%GPU%

