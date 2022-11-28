def get_input_image_names(list_names, directory_name, if_train=True):
    list_images = []
    list_masks = []
    list_test_ids = []

    for file_names in list_names['name']:
        name_red = 'red_' + file_names
        name_green = 'green_' + file_names
        name_blue = 'blue_' + file_names
        name_nir = 'nir_' + file_names

        if if_train:
            directory_type_name = "train"
            file_image = []
            name_mask = 'gt_' + file_names
            list_masks.append(f"{directory_name}/train_gt/{name_mask}.TIF")

        else:
            directory_type_name = "test"
            file_image = []
            list_test_ids.append(f"{file_names}.TIF")

        file_image.append(f"{directory_name}/{directory_type_name}_red/{name_red}.TIF")
        file_image.append(f"{directory_name}/{directory_type_name}_green/{name_green}.TIF")
        file_image.append(f"{directory_name}/{directory_type_name}_blue/{name_blue}.TIF")
        file_image.append(f"{directory_name}/{directory_type_name}_nir/{name_nir}.TIF")

        list_images.append(file_image)

    if if_train:
        return list_images, list_masks
    else:
        return list_images, list_test_ids
