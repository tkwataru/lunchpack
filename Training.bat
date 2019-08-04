set BCHSIZE=50
set EPOCH=1000
set PARA=8
set GPU=0
set WEIGHT=1
set TLIST=ng_pair_list_180408.txt
set VLIST=ng_pair_list_180408.txt
set OUT=result_180408
set RESUME=%OUT%\snapshot_latest

python trainLunchpack.py %TLIST% %VLIST% -o=%OUT% -B=%BCHSIZE% -b=%BCHSIZE% -e=%EPOCH% -g=%GPU% -j=%PARA% -w=%WEIGHT%
@rem python trainLunchpack.py %TLIST% %VLIST% -o=%OUT% -B=%BCHSIZE% -b=10 -e=%EPOCH% -g=%GPU% -j=%PARA% -w=%WEIGHT% --resume=%RESUME%


exit /b
