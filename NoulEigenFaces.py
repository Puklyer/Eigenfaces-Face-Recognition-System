import numpy as np
import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from numpy import linalg as la
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import random
import time
import pandas as pd


def nn_algorithm(A, search_image, norm='Euclidean'):
    z = np.zeros(A.shape[1])
    for i in range(A.shape[1]):
        if norm == 'Manhattan':
            z[i] = la.norm(A[:, i] - search_image, ord=1)
        elif norm == 'Euclidean':
            z[i] = la.norm(A[:, i] - search_image, ord=2)
        elif norm == 'Infinity':
            z[i] = la.norm(A[:, i] - search_image, ord=np.inf)
        elif norm == 'Cosine':
            z[i] = 1 - np.dot(A[:, i], search_image) / (la.norm(A[:, i]) * la.norm(search_image))

    min_position = np.argmin(z)
    return min_position


def lanczos_algorithm(A, k):
    m, n = A.shape
    q = np.zeros((m, k + 2))
    q[:, 1] = np.ones(m)
    q[:, 1] /= np.linalg.norm(q[:, 1])

    beta = 0
    q_prev = np.zeros(m)

    for i in range(1, k + 1):
        w = A.T @ (A @ q[:, i]) - beta * q_prev
        alpha = np.dot(w, q[:, i])
        w -= alpha * q[:, i]
        beta = np.linalg.norm(w)

        if beta == 0:
            break

        q_prev = q[:, i]
        q[:, i + 1] = w / beta

    print("m = ", m, "n = ", n)
    return q[:, 2:k + 2]  # Return only the first k vectors


def generate_data_matrix(base_folder, image_width=112, image_height=92, num_people=40, num_images_per_person=8):
    data_matrix = np.zeros((image_width * image_height, num_people * num_images_per_person))

    column_index = 0
    for person_index in range(num_people):
        person_folder = os.path.join(base_folder, f's{person_index + 1}')
        image_files = sorted(os.listdir(person_folder))[:num_images_per_person]
        for image_file in image_files:
            image_path = os.path.join(person_folder, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (image_width, image_height))
            image_vector = image.reshape(-1)
            data_matrix[:, column_index] = image_vector
            column_index += 1

    return data_matrix


def preprocEig(A, k):
    mean = np.mean(A, axis=1)
    A_centered = A - mean.reshape(-1, 1)
    L = A_centered.T @ A_centered
    eigenvalues, eigenvectors_L = np.linalg.eig(L)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors_L[:, sorted_indices[:k]]
    HQPB = A_centered @ top_eigenvectors
    HQPB = HQPB / np.linalg.norm(HQPB, axis=0)
    projections = A_centered.T @ HQPB
    return mean, HQPB, projections


def preprocEig_lanczos(A, k):
    A_centered = A - np.mean(A, axis=1).reshape(-1, 1)
    L = A_centered.T @ A_centered
    Q_lanczos = lanczos_algorithm(L, k)
    HQPB = A_centered @ Q_lanczos
    HQPB = HQPB / np.linalg.norm(HQPB, axis=0)
    projections = A_centered.T @ HQPB
    return HQPB, projections


def generate_class_representatives(A, num_people, num_images_per_person, method='random'):
    RC = np.zeros((A.shape[0], num_people))
    random_images = np.zeros(num_people, dtype=int)

    for person in range(num_people):
        random_images[person] = random.randint(person * num_images_per_person, (person + 1) * num_images_per_person - 1)
        RC[:, person] = A[:, random_images[person]]

    return RC


def preprocEig_RC(A, RC, k):
    mean_RC = np.mean(RC, axis=1)
    RC_centered = RC - mean_RC.reshape(-1, 1)
    L = RC_centered.T @ RC_centered
    eigenvalues, eigenvectors_L = np.linalg.eigh(L)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors_L[:, sorted_indices[:k]]
    HQPB = RC_centered @ top_eigenvectors
    HQPB = HQPB / np.linalg.norm(HQPB, axis=0)

    projections = A.T @ HQPB

    return mean_RC, HQPB, projections


image_width, image_height = 92, 112
num_people, num_images_per_person = 40, 8
base_folder = 'ORL'
data_matrix = generate_data_matrix(base_folder, image_width, image_height, num_people, num_images_per_person)

mean, HQPB, projections = None, None, None
test_image, test_vector = None, None


def update_eigenfaces(k_value, method='Standard'):
    global mean, HQPB, projections

    if method == 'Lanczos':
        HQPB, projections = preprocEig_lanczos(data_matrix, k=k_value)
    elif method == 'Classes':
        mean, HQPB, projections = preprocEig_RC(data_matrix, k=k_value)
    else:
        mean, HQPB, projections = preprocEig(data_matrix, k=k_value)

    messagebox.showinfo("Update", f"Principal components have been updated for k={k_value} with {method} method!")


def display_ghosts():
    global test_vector, HQPB
    if test_vector is None:
        messagebox.showerror("Error", "No image has been loaded!")
        return

    k = HQPB.shape[1]
    test_projections = test_vector @ HQPB

    num_ghosts = 10
    num_pages = (k + num_ghosts - 1) // num_ghosts

    for page in range(num_pages):
        start = page * num_ghosts
        end = min(start + num_ghosts, k)

        plt.figure(figsize=(15, 6))
        for i, index in enumerate(range(start, end)):
            ghost = test_projections[index] * HQPB[:, index]
            ghost = ghost.reshape(image_height, image_width)
            plt.subplot(1, end - start, i + 1)
            plt.imshow(ghost, cmap='gray')
            plt.title(f"Ghost {index + 1}")
            plt.axis('off')

        plt.suptitle(f"Ghosts {start + 1} - {end}")
        plt.show()


def recognize_face():
    global mean, HQPB, projections, test_image, test_vector
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.bmp;*.pgm")])
    if not file_path:
        return

    test_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if test_image is None:
        messagebox.showerror("Error", "Could not load the image. Check the file format!")
        return

    test_image = cv2.resize(test_image, (image_width, image_height))
    test_vector = test_image.reshape(-1)

    if mean is not None:
        test_vector = test_vector.astype(np.float64) - mean

    pr_test = test_vector @ HQPB
    min_position = nn_algorithm(projections.T, pr_test)
    person_index = min_position // num_images_per_person + 1
    found_image = data_matrix[:, min_position].reshape(image_height, image_width)
    test_image_tk = ImageTk.PhotoImage(image=Image.fromarray(test_image))
    found_image_tk = ImageTk.PhotoImage(image=Image.fromarray(found_image))

    top = tk.Toplevel()
    top.title("Recognition Results")

    tk.Label(top, text="Loaded image:").grid(row=0, column=0, padx=10, pady=10)
    tk.Label(top, text="Found image:").grid(row=0, column=1, padx=10, pady=10)

    tk.Label(top, image=test_image_tk).grid(row=1, column=0, padx=10, pady=10)
    tk.Label(top, image=found_image_tk).grid(row=1, column=1, padx=10, pady=10)

    top.test_image_tk = test_image_tk
    top.found_image_tk = found_image_tk

    messagebox.showinfo(
        "Result",
        f"The image is recognized as person: s{person_index}\nImage index: {min_position}"
    )


def recognize_face_RC(mean_RC, HQPB_RC, projections_RC, RC):
    global test_image, test_vector

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.bmp;*.pgm")])
    if not file_path:
        return

    test_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if test_image is None:
        messagebox.showerror("Error", "Could not load the image. Check the file format!")
        return

    test_image = cv2.resize(test_image, (image_width, image_height))

    test_vector = test_image.reshape(-1) - mean_RC

    pr_test = test_vector @ HQPB_RC

    # Calculate distances - find closest class representative
    distances = np.zeros(num_people)
    for i in range(num_people):
        distances[i] = np.linalg.norm(projections_RC[i, :] - pr_test)

    min_position = np.argmin(distances)
    person_index = min_position + 1

    found_image = RC[:, min_position].reshape(image_height, image_width)

    test_image_tk = ImageTk.PhotoImage(image=Image.fromarray(test_image))
    found_image_tk = ImageTk.PhotoImage(image=Image.fromarray(found_image))

    top = tk.Toplevel()
    top.title("Recognition Results with Representatives")

    tk.Label(top, text="Loaded image:").grid(row=0, column=0, padx=10, pady=10)
    tk.Label(top, text="Found representative image:").grid(row=0, column=1, padx=10, pady=10)

    tk.Label(top, image=test_image_tk).grid(row=1, column=0, padx=10, pady=10)
    tk.Label(top, image=found_image_tk).grid(row=1, column=1, padx=10, pady=10)

    top.test_image_tk = test_image_tk
    top.found_image_tk = found_image_tk

    print(f"Found class: s{person_index}, Image index in RC: {min_position}")

    messagebox.showinfo(
        "Result",
        f"The image is recognized as class: s{person_index}\nIndex in RC: {min_position}"
    )


def statistics(A, base_folder, image_width, image_height, num_people, num_images_per_person):
    time_preproc = {"Standard": {}, "Classes": {}, "Lanczos": {}}
    rr_results = {"Standard": {}, "Classes": {}, "Lanczos": {}}
    aqt_results = {"Standard": {}, "Classes": {}, "Lanczos": {}}
    total_time_results = {"Standard": {}, "Classes": {}, "Lanczos": {}}
    RC = generate_class_representatives(data_matrix, num_people, num_images_per_person, method='random')
    for method in ["Standard", "Classes", "Lanczos"]:
        print(f"Calculating statistics for {method} method...")

        for k in [20, 40, 60, 80, 100]:
            print(f"  Preprocessing for k={k}")
            t0 = time.time()
            if method == "Standard":
                mean, HQPB, projections = preprocEig(A, k)
            elif method == "Lanczos":
                HQPB, projections = preprocEig_lanczos(A, k)
            else:
                mean, HQPB, projections = preprocEig_RC(A, RC, k)
            t1 = time.time()
            time_preproc[method][k] = t1 - t0

            rr_values = {}
            aqt_values = {}

            for norm in ['Euclidean', 'Manhattan', 'Cosine', 'Infinity']:
                print(f"    NN Algorithm with {norm} norm")
                rr, aqt, t_total = calculate_rr_and_aqt(projections.T, norm=norm, algorithm='NN', k=1,
                                                        base_folder=base_folder,
                                                        image_width=image_width,
                                                        image_height=image_height,
                                                        num_people=num_people,
                                                        num_images_per_person=num_images_per_person,
                                                        mean=mean, HQPB=HQPB)
                rr_values[norm] = rr
                aqt_values[norm] = aqt
                total_time_results[method][k] = t_total

            rr_results[method][k] = rr_values
            aqt_results[method][k] = aqt_values

    return time_preproc, rr_results, aqt_results, total_time_results


def calculate_rr_and_aqt(A, norm='Euclidean', algorithm='NN', k=1, base_folder=None, image_width=112, image_height=92,
                         num_people=40, num_images_per_person=8, mean=mean, HQPB=HQPB):
    t0 = time.time()
    total_time = 0
    correct_count = 0

    for i in range(num_people):
        for img_num in [9, 10]:  # Assume images 9 and 10 are test images
            person_folder = os.path.join(base_folder, f's{i + 1}')
            image_path = os.path.join(person_folder, f'{img_num}.pgm')

            test_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            test_image = cv2.resize(test_image, (image_width, image_height))
            search_image = test_image.reshape(-1)
            search_image = search_image.T - mean
            search_image = search_image @ HQPB

            t1 = time.time()

            i0 = nn_algorithm(A, search_image, norm=norm)

            duration = time.time() - t1
            p0 = i0 // num_images_per_person + 1

            if p0 == (i + 1):
                correct_count += 1

            total_time += duration

    rr = correct_count / (num_people * 2)
    aqt = total_time / (num_people * 2)
    t_total = time.time() - t0

    return rr, aqt, t_total


def create_gui():
    root = tk.Tk()
    root.title("Face Recognition - Eigenfaces")
    root.geometry("500x600")

    title_label = tk.Label(root, text="Face Recognition", font=("Arial", 18, "bold"))
    title_label.pack(pady=10)

    k_label = tk.Label(root, text="Select k value:")
    k_label.pack(pady=5)

    k_combobox = ttk.Combobox(root, values=[20, 40, 60, 80, 100])
    k_combobox.pack(pady=5)
    k_combobox.current(4)

    preproc_label = tk.Label(root, text="Select preprocessing type:")
    preproc_label.pack(pady=5)

    preproc_combobox = ttk.Combobox(root, values=["Standard", "Lanczos"])
    preproc_combobox.pack(pady=5)
    preproc_combobox.current(0)

    def on_k_change():
        selected_k = int(k_combobox.get())
        preproc_method = preproc_combobox.get()

        if preproc_method == "Lanczos":
            update_eigenfaces(selected_k, method='Lanczos')
        else:
            update_eigenfaces(selected_k, method='Standard')

    update_k_button = ttk.Button(root, text="Update k", command=on_k_change)
    update_k_button.pack(pady=10)

    RC = generate_class_representatives(data_matrix, num_people, num_images_per_person, method='mean')
    mean_RC, HQPB_RC, projections_RC = preprocEig_RC(data_matrix, RC, k=100)

    represent_button = ttk.Button(
        root,
        text="Recognition with representatives",
        command=lambda: recognize_face_RC(mean_RC, HQPB_RC, projections_RC, RC)
    )
    represent_button.pack(pady=20)

    recognize_button = ttk.Button(root, text="Load image", command=recognize_face)
    recognize_button.pack(pady=20)

    ghosts_button = ttk.Button(root, text="Display ghosts", command=display_ghosts)
    ghosts_button.pack(pady=20)

    def run_statistics_and_save():
        print("Running statistics...")
        time_preproc, rr_results, aqt_results, total_time_results = statistics(
            data_matrix, base_folder, image_width, image_height, num_people, num_images_per_person
        )

        rr_df = []
        aqt_df = []

        for method, results in rr_results.items():
            for k, rr_norms in results.items():
                for norm, rr in rr_norms.items():
                    rr_df.append({'Method': method, 'k': k, 'Norm': norm, 'RR': rr})

        for method, results in aqt_results.items():
            for k, aqt_norms in results.items():
                for norm, aqt in aqt_norms.items():
                    aqt_df.append({'Method': method, 'k': k, 'Norm': norm, 'AQT': aqt})

        rr_df = pd.DataFrame(rr_df)
        aqt_df = pd.DataFrame(aqt_df)

        rr_df.to_csv('RecognitionRateEig.csv', index=False)
        aqt_df.to_csv('AQT_eig.csv', index=False)

        print("Statistics saved to RecognitionRateEig.csv and AQT_eig.csv files!")

    statistics_button = ttk.Button(
        root,
        text="Run statistics and save results",
        command=run_statistics_and_save
    )
    statistics_button.pack(pady=20)

    exit_button = ttk.Button(root, text="Exit", command=root.quit)
    exit_button.pack(pady=10)

    root.mainloop()


# Start GUI
create_gui()