mkdir raw_data
wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P raw_data/
tar -xvzf raw_data/cifar-10-python.tar.gz -C raw_data/
python sample_clean_class.py
python sample_attacked_class.py
