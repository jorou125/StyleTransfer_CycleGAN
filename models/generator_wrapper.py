# from implementations.cyclegan_pix2pix.models import create_model

# def get_model(is_custom=True):
#     """
#     Returns either the custom model or the one from the paper.
#     Useful for testing against the paper's results.
#     """
#     model = None

#     if is_custom:
#         pass
#     else:
#         opt = {}
#         model = create_model(opt)

from enum import Enum
import config
from models import generator as custom_generator
from models import discriminator as custom_discriminator
import implementations.rp_implementation.cyclegan as rp_cyclegan
import implementations.rp_implementation.train as rp_train

def get_models(implementation: config.Implementation):
    if implementation == config.Implementation.PAPER:
        pass # TODO
    elif implementation == config.Implementation.CUSTOM:
        gen_X = custom_generator.Generator().to(config.DEVICE)
        gen_Y = custom_generator.Generator().to(config.DEVICE)
        dis_X = custom_discriminator.Discriminator().to(config.DEVICE)
        dis_Y = custom_discriminator.Discriminator().to(config.DEVICE)

        return gen_X, gen_Y, dis_X, dis_Y
    elif implementation == config.Implementation.RP:
        Gen_AB = rp_cyclegan.GeneratorResNet(rp_train.input_shape, rp_train.hp.num_residual_blocks)
        Gen_BA = rp_cyclegan.GeneratorResNet(rp_train.input_shape, rp_train.hp.num_residual_blocks)

        Disc_A = rp_cyclegan.Discriminator(rp_train.input_shape)
        Disc_B = rp_cyclegan.Discriminator(rp_train.input_shape)

        if rp_train.cuda:
            Gen_AB = Gen_AB.cuda()
            Gen_BA = Gen_BA.cuda()
            Disc_A = Disc_A.cuda()
            Disc_B = Disc_B.cuda()

        ##############################################
        # Initialize weights
        ##############################################

        Gen_AB.apply(rp_train.initialize_conv_weights_normal)
        Gen_BA.apply(rp_train.initialize_conv_weights_normal)

        Disc_A.apply(rp_train.initialize_conv_weights_normal)
        Disc_B.apply(rp_train.initialize_conv_weights_normal)

        return Gen_AB, Gen_BA, Disc_A, Disc_B