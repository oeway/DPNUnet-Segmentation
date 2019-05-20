<docs lang="markdown">
imjoy cells image segementation
</docs>

<config lang="json">
{
  "name": "cells-segementation",
  "type": "native-python",
  "version": "0.1.0",
  "description": "[TODO: describe this plugin with one sentence.]",
  "tags": [],
  "ui": "",
  "cover": "",
  "inputs": null,
  "outputs": null,
  "flags": [],
  "icon": "extension",
  "api_version": "0.1.5",
  "env": "",
  "requirements": [],
  "dependencies": []
}
</config>

<script lang="python">
import os
print("os.getcwd():", os.getcwd())
os.chdir('imjoy-segmentation')
# from src.imgseg import segmentationUtils
# from src.imgseg import annotationUtils
# from src.imgseg import DRFNStoolbox
from src.geojson_utils import masks_to_annotation, gen_mask_from_geojson
# from imjoy import api



class ImJoyPlugin():
    def __init__(self):
        pass

    async def run(self, my=None):
        data_dialog = await api.showFileDialog(root=os.getcwd(), type="directory")
        datasets_dir = data_dialog.get("path")
        print("datasets_dir:", datasets_dir)
        for file_id in os.listdir(os.path.join(datasets_dir, "train")):
            file_path = os.path.join(datasets_dir, "train", file_id, "annotation.json")
            try:
                gen_mask_from_geojson([file_path], masks_to_create_value=["weighted_boarder"])
            except:
                print("generate mask error:", os.path.join(datasets_dir, "train", file_id))
        pass

    def setup(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass


api.export(ImJoyPlugin())
</script>

