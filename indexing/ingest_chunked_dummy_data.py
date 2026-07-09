import time
from typing import List, cast
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_text_splitters import RecursiveCharacterTextSplitter

from indexing.ingester import get_embedding_model

# Load environment variables
load_dotenv()

def get_embedding(text: str) -> List[float]:
    """Generate vector embedding for a given text chunk."""
    return cast(List[float], get_embedding_model().embed_query(text))

def ingest_data():
    # Initialize Elasticsearch client
    es = Elasticsearch(
        ["http://localhost:9200"],
        verify_certs=False,
        request_timeout=30
    )

    index_name = "research_papers"

    # Dummy data: 3 papers about 3D Gaussian Splatting and image quality metrics
    papers = [
        {
            "id": "paper_01",
            "title": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
            "author": "Kerbl et al.",
            "year": 2023,
            "source_file": "kerbl_2023.pdf",
            "content": """
            Radiance Field methods have recently revolutionized photorealistic synthesis of scenes captured with a handful of photos or videos.
            However, achieving high visual quality still requires neural networks that are costly to train and render, while recent faster methods inevitably trade off visual quality for speed.
            We introduce three key elements that allow us to achieve state-of-the-art visual quality while maintaining competitive training times and importantly allow high-quality real-time (>= 30 fps) novel-view synthesis at 1080p resolution.

            First, starting from sparse points produced during camera calibration, we represent the scene with 3D Gaussians that preserve desirable properties of continuous volumetric radiance fields for scene optimization while avoiding unnecessary computation in empty space.
            Second, we perform interleaved optimization/density control of the 3D Gaussians, notably optimizing anisotropic covariance to create an accurate representation of the scene.
            Third, we develop a fast GPU-based tile-based rasterizer that allows real-time rendering and significantly speeds up optimization.
            """
        },
        {
            "id": "paper_02",
            "title": "A Comparative Study of Image Quality Metrics: LPIPS vs MS-SSIM",
            "author": "Zhang, R.",
            "year": 2024,
            "source_file": "quality_metrics_study.pdf",
            "content": """
            Evaluating the perceptual quality of generated images is a critical task in computer vision.
            Traditional metrics like Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) often fail to capture human-like judgment of image quality, especially in the context of deep learning-based synthesis.
            Multi-Scale Structural Similarity (MS-SSIM) attempts to improve upon SSIM by evaluating image quality at multiple scales, but it still struggles with texture-heavy or highly detailed synthetic images.

            The Learned Perceptual Image Patch Similarity (LPIPS) metric has emerged as a powerful alternative.
            By leveraging deep neural network activations, LPIPS provides a distance measure that correlates much better with human perception.
            This study compares LPIPS and MS-SSIM across various datasets, including 3D Gaussian Splatting results, to determine which metric better aligns with visual fidelity in neural rendering tasks.
            """
        },
        {
            "id": "paper_03",
            "title": "Advancements in Perception-Aware Neural Rendering",
            "author": "Chen, L.",
            "year": 2025,
            "source_file": "neural_rendering_2025.pdf",
            "content": """
            Neural rendering has seen rapid growth, with a focus on improving both speed and visual fidelity.
            One of the key challenges remains the optimization of these models using loss functions that reflect human visual systems.
            Recent work has integrated perceptual losses like LPIPS directly into the training loops of 3D Gaussian Splatting architectures.

            By incorporating multi-scale perceptual feedback, we can guide the optimization of Gaussian primitives to better represent complex textures and lighting effects.
            Furthermore, the use of MS-SSIM as a complementary loss component helps maintain structural integrity.
            Our experiments show that a hybrid loss function combining LPIPS, MS-SSIM, and L1 loss leads to significantly more realistic novel-view synthesis with fewer artifacts.
            """
        }
    ]

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )

    total_chunks = 0

    for paper in papers:
        print(f"Processing: {paper['title']}")

        # Split text into chunks
        chunks = text_splitter.split_text(paper['content'])

        for i, chunk_text in enumerate(chunks):
            # Generate embedding
            embedding = get_embedding(chunk_text)

            # Use deterministic ID to prevent duplicates
            doc_id = f"{paper['id']}_chunk_{i}"

            # Prepare document for indexing
            doc = {
                "title": paper['title'],
                "content": chunk_text,
                "embedding": embedding,
                "metadata": {
                    "author": paper['author'],
                    "year": paper['year'],
                    "source_file": paper['source_file'],
                    "parent_id": paper['id'],
                    "chunk_id": i,
                    "page_number": 1
                }
            }

            # Index into Elasticsearch with explicit ID
            es.index(index=index_name, id=doc_id, document=doc)
            total_chunks += 1

        # Small delay to respect potential rate limits
        time.sleep(1)

    print(f"\nFinished indexing. Total chunks indexed: {total_chunks}")

if __name__ == "__main__":
    try:
        ingest_data()
    except Exception as e:
        print(f"Error during ingestion: {e}")
