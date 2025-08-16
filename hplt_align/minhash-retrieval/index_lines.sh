F=$1
zstdcat $F | cat -n | sed -r 's!^\s*([0-9]+)\s*\{!{"index":"'$F':\1",!' 
