python src/save_base.py
echo "Starting compression"
tar --use-compress-program='pigz --fast' -cf /build/subs/submission.tar.gz --dereference -C /build model -C /build/src main.py -C / libs
echo "Compression done successfully"
echo $(ls /build/subs)