import dataset_load
import os

# Parameters
pickle_name = "cmu_mosei_aligned"
pickle_folder = "cmu_mosei/pickle_files/"
align_text = True
align_label = True


def main():
    pickle_path = os.path.join(pickle_folder, pickle_name + ".pkl")

    if not os.path.exists(pickle_path):
        dataset_load.download_dataset(pickle_name,
                                      pickle_folder=pickle_folder,
                                      align_text=align_text,
                                      align_label=align_label)

    cmu_mosei = dataset_load.load_dataset_pickle(pickle_name,
                                                 pickle_folder=pickle_folder)


if __name__ == "__main__":
    main()
