python src/save_base.py
tar --use-compress-program='pigz --fast' -cf submission.tar.gz --dereference -C model . -C src main.py -C / libs