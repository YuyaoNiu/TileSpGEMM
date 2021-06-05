input="2757-matrix-AA-rtx2080.csv"

{
  read
  i=1
  while IFS=',' read -r id group name rows cols nonzeros
  do
    echo "$i $id $group $name $rows $cols $nonzeros"
    echo "~/UFget/MM/$group/$name/$name.mtx"
    ./test-aa -d 1 ~/UFget/MM/$group/$name/$name.mtx
    i=`expr $i + 1`
  done 
} < "$input"
