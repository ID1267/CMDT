def set_template(args):
    # Set the templates here

    if args.template.find('CMDT')>=0:
        args.input_setting='Y'
        args.input_mask='Phi_PhiPhiT'
        args.scheduler='CosineAnnealingLR'
        args.max_epoch=600000
