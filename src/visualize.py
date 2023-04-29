import base64
import io
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_top_features(top_features, title, name):
    fig, ax = plt.subplots()
    top_features.plot(kind="bar", x="feature", y="importance", legend=None, ax=ax)
    ax.set_ylabel("Importance")
    ax.set_xlabel("Features")
    ax.set_title(title)

    # Save the plot as an image file and return its path

    # Create the directory if it does not exist
    os.makedirs("src/static/images", exist_ok=True)
    
    image_path = f"images/{name}_importances.png"
    plt.savefig(f"src/static/{image_path}", bbox_inches="tight")
    plt.close(fig)

    return image_path
