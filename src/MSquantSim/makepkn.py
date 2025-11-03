import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def plot_indra(file_path, node_color='skyblue', edge_color='red', figsize=(8, 6)):
    """
    Reads an INDRA TSV file, creates a Prior Knowledge Network (PKN) as a directed graph, and plots it.

    Parameters:
    - file_path (str): Path to the INDRA TSV file.
    - node_color (str): Color of the graph nodes. Default is 'skyblue'.
    - edge_color (str): Color of the edge labels. Default is 'red'.
    - figsize (tuple): Figure size for the plot. Default is (8, 6).

    Returns:
    - G (nx.DiGraph): A directed graph object created from the TSV file.

    Raises:
    - FileNotFoundError: If the file is not found at the specified path.
    - pd.errors.EmptyDataError: If the file is empty.
    - pd.errors.ParserError: If there is a parsing error.
    """
    try:
        # Read the structure information from the TSV file
        pkn = pd.read_csv(file_path, sep='\t')
        print(f"INDRA data loaded successfully from {file_path}")

        # Create a directed graph
        G = nx.from_pandas_edgelist(pkn, source='source_hgnc_symbol', target='target_hgnc_symbol',
                                    create_using=nx.DiGraph())

        # Plot the graph
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G)  # Layout algorithm for better spacing
        nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_color, font_size=10, font_weight='bold',
                arrows=True)

        # If the graph has weights on edges, add edge labels
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color=edge_color)

        # Set title and show plot
        plt.title("Prior Knowledge Network from INDRA")
        plt.show()

        return G

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty.")
        raise
    except pd.errors.ParserError:
        print(f"Error: Parsing error encountered. Check the file format.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
    return


def plot_learned_structure(file_path, node_color='skyblue', edge_color='red', figsize=(8, 6)):
    """
    Reads a CSV file containing structure information, creates a directed graph, and plots it.

    Parameters:
    - file_path (str): Path to the CSV file.
    - node_color (str): Color of the graph nodes. Default is 'skyblue'.
    - edge_color (str): Color of the edge labels. Default is 'red'.
    - figsize (tuple): Figure size for the plot. Default is (8, 6).
    """
    structure_info = pd.read_csv(file_path)
    G = nx.from_pandas_edgelist(structure_info, source='from', target='to', create_using=nx.DiGraph())

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G)  # Layout algorithm for better spacing
    nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_color, font_size=10, font_weight='bold',
            arrows=True)

    # If the graph has weights on edges, add edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color=edge_color)

    plt.title("Learned Structure")
    plt.show()

