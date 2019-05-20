import os
from skimage import io
import shutil
# import segmentationUtils
# import annotationUtils
from imgseg import segmentationUtils
from imgseg import annotationUtils
from geojson import FeatureCollection, dump
from skimage import measure


def masks_to_annotation(datasets_dir, save_path):
    masks_dir = os.path.join(datasets_dir, "labels_all")
    nucleis_dir = os.path.join(datasets_dir, "images_all")

    # %% Process one folder and save as one json file allowing multiple annotation types
    simplify_tol = 0  # Tolerance for polygon simplification with shapely (0 to not simplify)

    # outputs_dir = os.path.abspath(os.path.join('..', 'data', 'postProcessing', 'mask2json'))
    if os.path.exists(masks_dir):
        print(f'Analyzing folder:{masks_dir}')
        for file in [f for f in os.listdir(masks_dir)]:
            file_id = os.path.splitext(file)[0]

            # Read png with mask
            print(f'Analyzing file:{file}')

            file_full = os.path.join(masks_dir, file)
            mask_img = io.imread(file_full)
            print("mask_img.shape:", mask_img.shape)
            mask = measure.label(mask_img)
            label = "nuclei"
            print("label:", label)
            sample_path = os.path.join(save_path, file_id)
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)
            io.imsave(os.path.join(sample_path, "mask.png"), mask_img)
            shutil.copyfile(os.path.join(nucleis_dir, file.replace(".tif", ".png")),
                            os.path.join(sample_path, "nuclei.png"))
            segmentationUtils.masks_to_polygon(mask, label=label, simplify_tol=simplify_tol,
                                               save_name=os.path.join(sample_path, "annotation.json"))


if __name__ == "__main__":
    datasets_dir = "/home/alex/Downloads/test/data"
    # masks_dir = os.path.join(datasets_dir, "labels_all")
    # nucleis_dir = os.path.join(datasets_dir, "labels_all")
    masks_to_annotation(datasets_dir, "/home/alex/Downloads/test/data/anet/train")
    # print(os.path.splitext("annotation.json"))


