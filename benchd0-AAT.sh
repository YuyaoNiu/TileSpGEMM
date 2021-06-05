input="2757-matrix-AAT-titanrtx.csv"

{
  read
  i=1
  while IFS=',' read -r id group name rows cols nonzeros
  do
    echo "$i $id $group $name $rows $cols $nonzeros"
    echo "~/UFget/MM/$group/$name/$name.mtx"
    ./test-aat -d 0 ~/UFget/MM/$group/$name/$name.mtx
    i=`expr $i + 1`
  done 
} < "$input"
