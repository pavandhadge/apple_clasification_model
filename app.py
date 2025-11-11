from ui import main_ui


def main():
    try:
        import tensorflow as tf
        tf_available = True
    except ImportError:
        tf_available = False

    main_ui(tf_available)

if __name__ == "__main__":
    main()
