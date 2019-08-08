import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='sagan', choices=['sagan', 'qgan'])
    parser.add_argument('--adv_loss', type=str, default='wgan-gp', choices=['wgan-gp', 'hinge']) #对抗(adv)损失
    parser.add_argument('--imsize', type=int, default=128)
    parser.add_argument('--g_num', type=int, default=5) #TODO:
    parser.add_argument('--chn', type=int, default=64) #作为基数，所有层都是几倍于它的深度
    parser.add_argument('--z_dim', type=int, default=120) #必须是3的倍数，因为G的潜在层被分成三部分喂入残差块
    parser.add_argument('--g_conv_dim', type=int, default=64) #TODO: 这两个貌似没用，纬度用cnh实现
    parser.add_argument('--d_conv_dim', type=int, default=64) #TODO:
    parser.add_argument('--lambda_gp', type=float, default=10) #梯度惩罚项的系数
    parser.add_argument('--version', type=str, default='sagan_1') #TODO: 文件夹名字而已

    # Training setting
    parser.add_argument('--total_step', type=int, default=1000000, help='how many times to update the generator')
    parser.add_argument('--d_iters', type=float, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=12) #TODO:
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0004)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None) # TODO:输入之前已经训练的代数，在trainer.py/load_pretrained_model()中将根据这一数据找到对应代的文件并导入

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--gpus', type=str, default='0', help='gpuids eg: 0,1,2,3  --parallel True  ')
    parser.add_argument('--dataset', type=str, default='lsun', choices=['lsun', 'celeb','off'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--image_path', type=str, default='./data')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--attn_path', type=str, default='./attn')

    # Step size
    parser.add_argument('--log_step', type=int, default=10) #每隔多少step打印在控制台一条信息
    parser.add_argument('--sample_step', type=int, default=100) #每隔多少step采样一张图片
    parser.add_argument('--model_save_step', type=float, default=1.0)


    return parser.parse_args()
