# Trojan Attack

Code for trojan attack on 5 classes (['airplane','automobile','frog','cat','ship']) sampled from cifar10 dataset. Clean model indicates the model is trained on clean dataset and attacked model is trained on poisoned dataset. Potential solution of searching for the attacked model is performed by inspecting the visualization of activation response.    

# Usage
##  Evaluate pretrained model
1. Clone the project to directory 
```
git clone https://github.com/bill86416/trojan_attack.git
```
2. Initiate the conda environment
```
conda env create -f environment.yml -n trojan
conda activate trojan
```
3. Download the cifar10 dataset and generate attacked dataset (i.e. Some images in Frog class is mislabeled as Airplane class)
```
cd datasets
sh download_data.sh 
```
4. Train clean and attacked model
```
sh run.sh
```
5. Visualize the trojan image with clean attacked model
```
sh run_visualization.sh
```

# Ackowledgement
Please email to Chih-Hsing Ho (Bill) (bill86416@gmail.com) if further issues are encountered.
