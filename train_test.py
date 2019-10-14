from keras.layers import *
import keras.backend as K
import tensorflow as tf
import os
import cv2
import glob
import time
import numpy as np
from pathlib import PurePath, Path
from IPython.display import clear_output
from keras_vggface.vggface import VGGFace

from utils import showG, showG_mask, showG_eyes

model, vggface = (None, None)

def run(img_dirA, img_dirB, TOTAL_ITERS = 40000):
    global model, vggface
    
    img_dirA_bm_eyes = f"{img_dirA}/binary_masks_eyes2"
    img_dirB_bm_eyes = f"{img_dirB}/binary_masks_eyes2"

    # Path to saved model weights
    models_dir = "./models"

    # Number of CPU cores
    num_cpus = os.cpu_count()

    # Input/Output resolution
    RESOLUTION = 64 # 64x64, 128x128, 256x256
    assert (RESOLUTION % 64) == 0, "RESOLUTION should be 64, 128, or 256."

    # Batch size
    batchSize = 8
    assert (batchSize != 1 and batchSize % 2 == 0) , "batchSize should be an even number."

    # Use motion blurs (data augmentation)
    # set True if training data contains images extracted from videos
    use_da_motion_blur = False 

    # Use eye-aware training
    # require images generated from prep_binary_masks.ipynb
    use_bm_eyes = True

    # Probability of random color matching (data augmentation)
    prob_random_color_match = 0.5

    da_config = {
        "prob_random_color_match": prob_random_color_match,
        "use_da_motion_blur": use_da_motion_blur,
        "use_bm_eyes": use_bm_eyes
    }

    # Architecture configuration
    arch_config = {}
    arch_config['IMAGE_SHAPE'] = (RESOLUTION, RESOLUTION, 3)
    arch_config['use_self_attn'] = True
    arch_config['norm'] = "instancenorm" # instancenorm, batchnorm, layernorm, groupnorm, none
    arch_config['model_capacity'] = "standard" # standard, lite

    # VGGFace ResNet50
    vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))

    # Loss function weights configuration
    loss_weights = {}
    loss_weights['w_D'] = 0.1 # Discriminator
    loss_weights['w_recon'] = 1. # L1 reconstruction loss
    loss_weights['w_edge'] = 0.1 # edge loss
    loss_weights['w_eyes'] = 30. # reconstruction and edge loss on eyes area
    loss_weights['w_pl'] = (0.01, 0.1, 0.3, 0.1) # perceptual loss (0.003, 0.03, 0.3, 0.3)

    # Init. loss config.
    loss_config = {}
    loss_config["gan_training"] = "mixup_LSGAN" # "mixup_LSGAN" or "relativistic_avg_LSGAN"
    loss_config['use_PL'] = False
    loss_config["PL_before_activ"] = False
    loss_config['use_mask_hinge_loss'] = False
    loss_config['m_mask'] = 0.
    loss_config['lr_factor'] = 1.
    loss_config['use_cyclic_loss'] = False

    from networks.faceswap_gan_model import FaceswapGANModel
    model = FaceswapGANModel(**arch_config)
    model.load_weights(path=models_dir)

    from colab_demo.vggface_models import RESNET50
    vggface = RESNET50(include_top=False, weights=None, input_shape=(224, 224, 3))
    vggface.load_weights("rcmalli_vggface_tf_notop_resnet50.h5")

    #vggface.summary()

    model.build_pl_model(vggface_model=vggface, before_activ=loss_config["PL_before_activ"])
    model.build_train_functions(loss_weights=loss_weights, **loss_config)
    from data_loader.data_loader import DataLoader
    # Create ./models directory
    Path(f"models").mkdir(parents=True, exist_ok=True)

    # Get filenames
    faces_A = f"{img_dirA}/raw_faces"
    faces_B = f"{img_dirB}/raw_faces"

    train_A = glob.glob(f"{faces_A}/*.*")
    train_B = glob.glob(f"{faces_B}/*.*")

    train_AnB = train_A + train_B

    assert len(train_A), f"No image found in {faces_A}"
    assert len(train_B), f"No image found in {faces_B}"
    print ("Number of images in folder A: " + str(len(train_A)))
    print ("Number of images in folder B: " + str(len(train_B)))

    if use_bm_eyes:
        assert len(glob.glob(img_dirA_bm_eyes+"/*.*")), "No binary mask found in " + str(img_dirA_bm_eyes)
        assert len(glob.glob(img_dirB_bm_eyes+"/*.*")), "No binary mask found in " + str(img_dirB_bm_eyes)
        assert len(glob.glob(img_dirA_bm_eyes+"/*.*")) == len(train_A), \
        "Number of faceA images does not match number of their binary masks. Can be caused by any none image file in the folder."
        assert len(glob.glob(img_dirB_bm_eyes+"/*.*")) == len(train_B), \
        "Number of faceB images does not match number of their binary masks. Can be caused by any none image file in the folder."


    def show_loss_config(loss_config):
        for config, value in loss_config.items():
            print(f"{config} = {value}")

    def reset_session(save_path):
        global model, vggface
        global train_batchA, train_batchB
        model.save_weights(path=save_path)
        del model
        del vggface
        del train_batchA
        del train_batchB
        K.clear_session()
        model = FaceswapGANModel(**arch_config)
        model.load_weights(path=save_path)
        vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
        model.build_pl_model(vggface_model=vggface, before_activ=loss_config["PL_before_activ"])
        train_batchA = DataLoader(train_A, train_AnB, batchSize, img_dirA_bm_eyes,
                                  RESOLUTION, num_cpus, K.get_session(), **da_config)
        train_batchB = DataLoader(train_B, train_AnB, batchSize, img_dirB_bm_eyes, 
                                  RESOLUTION, num_cpus, K.get_session(), **da_config)

    # Start training
    t0 = time.time()

    # This try/except is meant to resume training that was accidentally interrupted
    try:
        gen_iterations
        print(f"Resume training from iter {gen_iterations}.")
    except:
        gen_iterations = 0

    errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
    errGAs = {}
    errGBs = {}
    # Dictionaries are ordered in Python 3.6
    for k in ['ttl', 'adv', 'recon', 'edge', 'pl']:
        errGAs[k] = 0
        errGBs[k] = 0

    display_iters = 300
    backup_iters = 5000

    global train_batchA, train_batchB
    train_batchA = DataLoader(train_A, train_AnB, batchSize, img_dirA_bm_eyes, 
                              RESOLUTION, num_cpus, K.get_session(), **da_config)
    train_batchB = DataLoader(train_B, train_AnB, batchSize, img_dirB_bm_eyes, 
                              RESOLUTION, num_cpus, K.get_session(), **da_config)

    clear_output()
    loss_config['use_PL'] = True
    loss_config['use_mask_hinge_loss'] = False
    loss_config['m_mask'] = 0.0
    reset_session(models_dir)
    print("Building new loss funcitons...")
    show_loss_config(loss_config)
    model.build_train_functions(loss_weights=loss_weights, **loss_config)
    print("Done.")

    while gen_iterations <= TOTAL_ITERS: 

        # Loss function automation
        if gen_iterations == (TOTAL_ITERS//5 - display_iters//2):
            clear_output()
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = False
            loss_config['m_mask'] = 0.0
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Done.")
        elif gen_iterations == (TOTAL_ITERS//5 + TOTAL_ITERS//10 - display_iters//2):
            clear_output()
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = True
            loss_config['m_mask'] = 0.5
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Complete.")
        elif gen_iterations == (2*TOTAL_ITERS//5 - display_iters//2):
            clear_output()
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = True
            loss_config['m_mask'] = 0.2
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Done.")
        elif gen_iterations == (TOTAL_ITERS//2 - display_iters//2):
            clear_output()
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = True
            loss_config['m_mask'] = 0.4
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Done.")
        elif gen_iterations == (2*TOTAL_ITERS//3 - display_iters//2):
            clear_output()
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = False
            loss_config['m_mask'] = 0.
            loss_config['lr_factor'] = 0.3
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Done.")
        elif gen_iterations == (8*TOTAL_ITERS//10 - display_iters//2):
            clear_output()
            model.decoder_A.load_weights("models/decoder_B.h5") # swap decoders
            model.decoder_B.load_weights("models/decoder_A.h5") # swap decoders
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = True
            loss_config['m_mask'] = 0.1
            loss_config['lr_factor'] = 0.3
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Done.")
        elif gen_iterations == (9*TOTAL_ITERS//10 - display_iters//2):
            clear_output()
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = False
            loss_config['m_mask'] = 0.0
            loss_config['lr_factor'] = 0.1
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Done.")

        if gen_iterations == 5:
            print ("working.")

        # Train dicriminators for one batch
        data_A = train_batchA.get_next_batch()
        data_B = train_batchB.get_next_batch()
        errDA, errDB = model.train_one_batch_D(data_A=data_A, data_B=data_B)
        errDA_sum +=errDA[0]
        errDB_sum +=errDB[0]

        # Train generators for one batch
        data_A = train_batchA.get_next_batch()
        data_B = train_batchB.get_next_batch()
        errGA, errGB = model.train_one_batch_G(data_A=data_A, data_B=data_B)
        errGA_sum += errGA[0]
        errGB_sum += errGB[0]
        for i, k in enumerate(['ttl', 'adv', 'recon', 'edge', 'pl']):
            errGAs[k] += errGA[i]
            errGBs[k] += errGB[i]
        gen_iterations+=1

        # Visualization
        if gen_iterations % display_iters == 0:
            clear_output()

            # Display loss information
            show_loss_config(loss_config)
            print("----------") 
            print('[iter %d] Loss_DA: %f Loss_DB: %f Loss_GA: %f Loss_GB: %f time: %f'
            % (gen_iterations, errDA_sum/display_iters, errDB_sum/display_iters,
               errGA_sum/display_iters, errGB_sum/display_iters, time.time()-t0))  
            print("----------") 
            print("Generator loss details:")
            print(f'[Adversarial loss]')  
            print(f'GA: {errGAs["adv"]/display_iters:.4f} GB: {errGBs["adv"]/display_iters:.4f}')
            print(f'[Reconstruction loss]')
            print(f'GA: {errGAs["recon"]/display_iters:.4f} GB: {errGBs["recon"]/display_iters:.4f}')
            print(f'[Edge loss]')
            print(f'GA: {errGAs["edge"]/display_iters:.4f} GB: {errGBs["edge"]/display_iters:.4f}')
            if loss_config['use_PL'] == True:
                print(f'[Perceptual loss]')
                try:
                    print(f'GA: {errGAs["pl"][0]/display_iters:.4f} GB: {errGBs["pl"][0]/display_iters:.4f}')
                except:
                    print(f'GA: {errGAs["pl"]/display_iters:.4f} GB: {errGBs["pl"]/display_iters:.4f}')

            # Display images
            print("----------") 
            wA, tA, _ = train_batchA.get_next_batch()
            wB, tB, _ = train_batchB.get_next_batch()
            print("Transformed (masked) results:")
            showG(tA, tB, model.path_A, model.path_B, batchSize)   
            print("Masks:")
            showG_mask(tA, tB, model.path_mask_A, model.path_mask_B, batchSize)  
            print("Reconstruction results:")
            showG(wA, wB, model.path_bgr_A, model.path_bgr_B, batchSize)           
            errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
            for k in ['ttl', 'adv', 'recon', 'edge', 'pl']:
                errGAs[k] = 0
                errGBs[k] = 0

            # Save models
            model.save_weights(path=models_dir)

        # Backup models
        if gen_iterations % backup_iters == 0: 
            bkup_dir = f"{models_dir}/backup_iter{gen_iterations}"
            Path(bkup_dir).mkdir(parents=True, exist_ok=True)
            model.save_weights(path=bkup_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='train model on 2 faces')
    parser.add_argument('a', metavar='a', type=str, help='source face A')
    parser.add_argument('b', metavar='b', type=str, help='source face B')
    parser.add_argument('--iters', metavar='iters', type=int, help='iterations', default=40000)

    args = parser.parse_args()

    run(args.a, args.b)
