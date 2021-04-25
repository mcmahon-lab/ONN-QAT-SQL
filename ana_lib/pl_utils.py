#Useful general functions

def default_scheduler(lr_scheduler):
    return {'scheduler':lr_scheduler, 'monitor':'val_checkpoint_on'}

def default_scheduler2(lr_scheduler):
    return {'scheduler':lr_scheduler, 'monitor':'asdfhakjsdhfjkahsdjfk'}