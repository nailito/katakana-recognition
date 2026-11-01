def split(metadata,training,validation,test):  

    if training + validation + test != 1:
        print("please make sure that the sum of the training, validation and test segment equals 1") 

        pass

    else:
        writer_list = np.random.default_rng(12).permutation(metadata["writer_id"].unique())
        writer_train = writer_list[0:int(len(writer_list)*training)]
        writer_valid = writer_list[int(len(writer_list)*training)-1:(int(len(writer_list)*training)+int(len(writer_list)*validation))]
        writer_test = writer_list[((int(len(writer_list)*training)+int(len(writer_list)*validation)))-1::]

        df_train = metadata[metadata["writer_id"].isin(writer_train)]
        df_valid = metadata[metadata["writer_id"].isin(writer_valid)]
        df_test = metadata[metadata["writer_id"].isin(writer_test)]

    return df_train, df_valid, df_test