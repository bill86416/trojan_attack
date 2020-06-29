from gradcam import *
import os
import argparse
import logging
import sys

parser = argparse.ArgumentParser(description='TroJanAI')
parser.add_argument('-e', '--epochs', action='store', default=40, type=int, help='epochs (default: 40)')
parser.add_argument('--batchSize', action='store', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', action='store', default=0.01, type=float, help='learning rate (default: 0.0001)')
parser.add_argument('--m', '--momentum', action='store', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--w', '--weight-decay', action='store', default=0, type=float, help='regularization weight decay (default: 0.0)')
parser.add_argument('--train_f', action='store_false', default=True, help='Flag to train (STORE_FALSE)(default: True)')
parser.add_argument('--gpu_num', action='store', default=0, type=int, help='gpu_num (default: 0)')
parser.add_argument("--net", default='vgg19', const='vgg19',nargs='?', choices=['resnet20', 'vgg19'], help="net model(default:vgg19)")
parser.add_argument('--trial', action='store', default=1, type=int, help='trial (default: 1)')
parser.add_argument('--attacked', action='store_true', default=False, help='Flag for whether model is attacked')
parser.add_argument('--image_id', type=int,  help='Input dir path')
parser.add_argument('--image_dir_path', type=str, default="./datasets/selected_attack_dataset/test_attacked_image/frog", help='Input dir path')
parser.add_argument('--output_dir_path', type=str, default="./visualization",  help='Output dir path')
arg = parser.parse_args()

if __name__ == '__main__':

    torch.cuda.set_device(arg.gpu_num)
    if not os.path.exists(arg.output_dir_path):
        os.makedirs(arg.output_dir_path)
        
    if arg.attacked:
        model_path = './model/model_'+arg.net + '_attacked_' + str(arg.trial)+'.pt'
    else:
        model_path = './model/model_'+arg.net + '_clean_' + str(arg.trial)+'.pt'
    
           
    model = vgg19().cuda()
    print("Loading model")
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    else:
        print("No pretrained model found")
        sys.exit()
    
    model.eval()
    grad_cam = GradCam(model,target_layer_names = ["10"], use_cuda=True)
  
    if arg.image_id is None:
        arg.image_id = int(os.listdir(arg.image_dir_path)[0][:-4])
    
    image_path = arg.image_dir_path + '/' + str(arg.image_id) + '.png'
    
    print("Reading image {}".format(image_path))
    img = cv2.imread(image_path, 1)
    if arg.attacked:
        filename = arg.output_dir_path + '/' + "frog_image_" + str(arg.image_id) +  '_attacked_'  + arg.net + '_trial_' +str(arg.trial)
    else:
        filename = arg.output_dir_path + '/' + "frog_image_" + str(arg.image_id) +  '_clean_'  + arg.net + '_trial_' + str(arg.trial)
    img = np.float32(cv2.resize(img, (32, 32))) / 255
    input = preprocess_image(img)


    mask = grad_cam(input, None)

    show_cam_on_image(img, mask, filename)

    gb_model = GuidedBackpropReLUModel(model, use_cuda=True)
    gb = gb_model(input, index=None)
    utils.save_image(torch.from_numpy(gb), filename + '_gb.png')

    cam_mask = np.zeros(gb.shape)
    for i in range(0, gb.shape[0]):
        cam_mask[i, :, :] = mask

    cam_gb = np.multiply(cam_mask, gb)
    utils.save_image(torch.from_numpy(cam_gb), filename + '_cam_gb.png')