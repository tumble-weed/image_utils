# x,y pyramids for gpnn
def create_pyramid2(mask_tensor,R,PATCH_SIZE,STRIDE):
    mask_pyramid = []
    i = 1
    query_pyramid = []
    key_pyramid = []
    query_ratios = OrderedDict()
    key_ratios = OrderedDict()
    #=================================================
    while True:
        complement_smaller = pyrdown(
            (1 - mask_tensor), 
            mask = (1-mask_tensor),
            border_type = 'reflect', 
            align_corners = True, 
            factor = R ** i).permute(0,2,3,1)
        smaller = 1 - complement_smaller
        smaller = (smaller != 0.).float();print(colored('setting the pyramid level to binary eary','yellow'))
        if min(smaller.shape[1:3]) < max(PATCH_SIZE):
            break
        mask_pyramid.append(smaller)
        if smaller.sum() == 0:
            break
        #====================================================
        # # standard pyramid:
        # patch_size = PATCH_SIZE
        # min_dim = min(smaller.shape[1:3])
        # if max(patch_size) > min_dim:
        #     patch_size = (min_dim,min_dim)
        # patches = extract_patches(smaller, patch_size, STRIDE)
        # mask_query = torch.all(patches, dim=-1)
        # mask_query = torch.all(mask_query, dim=-1)           
        # mask_key = mask_query
        # key_pyramid.append(mask_key)
        # query_pyramid.append(mask_query)
        #====================================================
        i += 1
    # import ipdb;ipdb.set_trace()
    #=================================================
    mask_pyramid.insert(0,mask_tensor.permute(0,2,3,1))
    
    print(colored('TODO:cutting off mask_pyramid at 8','red'))
    mask_pyramid = mask_pyramid[:8]
    n_levels = len(mask_pyramid)
    import ipdb;ipdb.set_trace()
    mask_pyramid2 = []
    for mask in mask_pyramid:
        pass
    for mask in ((mask_pyramid)):
        #====================================================
        # standard pyramid:
        # mask = (mask == 1.).float()
        if False:
            mask = (mask != 0.).float()
        patch_size = PATCH_SIZE
        min_dim = min(mask.shape[1:3])
        if max(patch_size) > min_dim:
            patch_size = (min_dim,min_dim)
        patches = extract_patches(mask, patch_size, STRIDE)
        mask_query = torch.any(patches, dim=-1)
        mask_query = torch.any(mask_query, dim=-1)           
        #=========================================
        # mask_key = mask_query
        mask_key = torch.any(patches, dim=-1)
        mask_key = torch.any(mask_key, dim=-1)                      
        #=========================================         
        key_pyramid.append(mask_key)
        query_pyramid.append(mask_query)
        #====================================================  
        available_area = np.prod(mask.shape[1:3])
        query_ratio  = (mask_query).sum()/available_area
        key_ratio = (mask_key).sum()/available_area
        query_ratios[available_area] = query_ratio.item()
        key_ratios[available_area] = key_ratio.item()
        print('see if ratios are fine')
        # import ipdb;ipdb.set_trace()
        #====================================================  
    print('TODO: will key query and mask pyramids be same?')
    print('TODO: mask_pyramid will have float, so cant be same')
    return mask_pyramid,query_pyramid,key_pyramid
self.mask_pyramid,self.query_pyramid,self.key_pyramid = create_pyramid2(self.mask_tensor,self.R,self.PATCH_SIZE,self.STRIDE)