def set_template(args):
    # Set the templates here
    if args.template.find('mst') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Phi'

    if args.template.find('gap_net') >= 0 or args.template.find('admm_net') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Phi_PhiPhiT'
        args.milestones = list(range(30,args.max_epoch,30))
        args.gamma = 0.9
        args.learning_rate = 1e-3

    if args.template.find('tsa_net') >= 0:
        args.input_setting = 'HM'
        args.input_mask = None

    if args.template.find('hdnet') >= 0:
        args.input_setting = 'H'
        args.input_mask = None

    if args.template.find('dgsmp') >= 0:
        args.input_setting = 'Y'
        args.input_mask = None
        args.batch_size = 2
        args.milestones = [args.max_epoch]    # fix the learning rate during training, following the official implementation
        args.learning_rate = 1e-4

    if args.template.find('birnat') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Phi'
        args.batch_size = 1
        args.max_epoch = 600000
        args.milestones = list(range(10,100,10))   # fix the learning rate during training, following the official implementation
        args.learning_rate = 1.5e-4

    if args.template.find('mst_plus_plus') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        args.scheduler = 'CosineAnnealingLR'
    
    if args.template.find('cst') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        args.scheduler = 'CosineAnnealingLR'
        args.max_epoch = 600000

    if args.template.find('dnu') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Phi_PhiPhiT'
        args.batch_size = 2
        args.max_epoch = 600000
        args.milestones = range(10,args.max_epoch,10)
        args.gamma = 0.9
        args.learning_rate = 4e-4

    if args.template.find('lambda_net') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Phi'
        args.learning_rate = 1.5e-4

    if args.template.find('dauhst')>=0:
        args.input_setting='Y'
        args.input_mask='Phi_PhiPhiT'
        args.scheduler='CosineAnnealingLR'
        args.max_epoch=600000

    if args.template.find('padut')>=0:
        args.input_setting='Y'
        args.input_mask='Phi_PhiPhiT'
        args.scheduler='CosineAnnealingLR'
        # args.max_epoch=600000

    if args.template.find('nocnn')>=0:
        args.input_setting='Y'
        args.input_mask='Phi_PhiPhiT'
        args.scheduler='CosineAnnealingLR'
        # args.max_epoch=50000
        args.max_epoch=600000
        # args.max_epoch=300
        # args.batch_size=9

    if args.template.find('nocnn_spat')>=0:
        args.input_setting='Y'
        args.input_mask='Phi_PhiPhiT'
        args.scheduler='CosineAnnealingLR'
        # args.max_epoch=50000
        args.max_epoch=600000
        # args.max_epoch=300
        # args.batch_size=9
