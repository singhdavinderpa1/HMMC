from optparse import OptionParser

#dataset = 'SCDB'

def hehe():
    #print(dataset)
    for x in [train_df, val_df]:
        print(x)


if __name__ == "__main__":

    parser = OptionParser()

    # paths
    parser.add_option('--data_path', default='Toy_Dataset_v4', dest="data_path", type="string", help='path to the data')

    # get variables
    (options, args) = parser.parse_args()
    print(options)
    global dataset_name
    dataset_name = options.data_path
    global train_df
    global val_df
    ##################
    if (dataset_name == 'Toy_Dataset_v4'):
        dataset_name = 'Toy_Dataset_v4'
        train_df = "train"
        val_df = 'val'
    elif (dataset_name == 'CelebA'):
        #global dataset_name
        #global dataset_name
        dataset_name = "CelebA/MySplit"
        train_df = 'train_gender'
        val_df = 'val_gender'
    else:
        train_df = "D7PH2_ISIC2019CuratedV2"
    print(dataset_name)
    print(f' dataset selected is {dataset_name} and train is {train_df} val is {val_df}')
