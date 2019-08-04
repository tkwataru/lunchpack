set OUT=./Segmentation_180407_All
@rem set MODEL=result_180408\model_latest
set MODEL=result_180408\model_epoch-0970
set GPU=0
set VLIST=all_lunchpack_list.txt

python inferLunchpack_img.py %VLIST% -R=%OUT% -m=%MODEL% -g=%GPU%

