import pandas as pd
from tqdm import tqdm
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid


def main():
    df = pd.read_parquet(r'C:\Users\Noel\Documents\THESIS\Feature Visualization\Visualization\test65_activations.parquet')
    layers_of_interest = [name for name in df.columns.tolist() if "conv" in name or "fc" in name]
    for layer in tqdm(layers_of_interest):
        for channel_n in tqdm(range(df[layer].values[0].shape[0])):

            assert channel_n <= len(df[layer]), f"Channel Index out of range. {layer} has {len(df[layer])} channels."
            column_name = f'activations_{layer}_channel_{channel_n}'
            df[column_name] = df[layer].map(lambda x:x[channel_n])

            df.sort_values(by=column_name, inplace=True, ascending=True)
            # df.head(10)

            top_images = df["path"][:10].tolist()
            images_to_disp = []
            for image_path in top_images:
                images_to_disp.append(read_image(image_path))

            grid = make_grid(images_to_disp, nrow=5, padding=0)

            img = torchvision.transforms.ToPILImage()(grid)
            # img.show()
            img.save(fp=rf"C:\Users\Noel\Documents\THESIS\Outputs_Feature_Visualization\test65outputs\{layer}\{channel_n}_Negative_Activations.jpg")
            df = df.drop(columns=[f'activations_{layer}_channel_{channel_n}'])


if __name__ == "__main__":
    main()
