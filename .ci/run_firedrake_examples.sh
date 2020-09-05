cd examples
for i in $(find . -name '*firedrake*.py' -exec grep -q __main__ '{}' \; -print ); do
    echo "-----------------------------------------------------------------------"
    echo "RUNNING $i"
    echo "-----------------------------------------------------------------------"
    dn=$(dirname "$i")
    bn=$(basename "$i")
    (cd $dn; time python3 "$bn")
done
