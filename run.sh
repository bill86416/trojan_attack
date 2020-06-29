python main.py --trial 1 --gpu_num 0 --net 'vgg19' 
python main.py --trial 1 --gpu_num 0 --net 'vgg19' --attacked  

python eval.py --trial 1 --gpu_num 0 --net 'vgg19' 
python eval.py --trial 1 --attacked --gpu_num 0 --net 'vgg19'   

python eval.py --trial 1 --gpu_num 0 --net 'vgg19' --data_type attacked_dataset 
python eval.py --trial 1 --attacked --gpu_num 0 --net 'vgg19'  --data_type attacked_dataset  

python eval.py --trial 1 --gpu_num 0 --net 'vgg19' --data_type attacked_class_only 
python eval.py --trial 1 --attacked --gpu_num 0 --net 'vgg19'  --data_type attacked_class_only  
