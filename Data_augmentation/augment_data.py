import os
import numpy as np
from pathlib import Path
from scipy.ndimage import map_coordinates
from skimage import io
from multiprocessing import Pool, cpu_count, Manager
import random
from skimage.color import rgb2gray

def generate_single_patch_displacement(patch_sizes, SubsetSize, desviacion=0.8):
    """
    Genera un desplazamiento aleatorio aplicado a un solo parche en la imagen.
    """
    patch_size = random.choice(patch_sizes)

    # Seleccionar aleatoriamente la posición del parche en la imagen
    max_x = SubsetSize - patch_size
    max_y = SubsetSize - patch_size
    patch_x = random.randint(0, max_x)
    patch_y = random.randint(0, max_y)

    # Crear matrices de desplazamiento llenas de ceros
    disp_x = np.zeros((SubsetSize, SubsetSize), dtype=np.float64)
    disp_y = np.zeros((SubsetSize, SubsetSize), dtype=np.float64)

    # Generar desplazamientos aleatorios para el parche seleccionado (siguiendo una distribución normal)
    f_patch = np.random.normal(loc=0, scale=desviacion, size=(patch_size, patch_size))
    g_patch = np.random.normal(loc=0, scale=desviacion, size=(patch_size, patch_size))

    # Limitar los desplazamientos al rango de [-3, 3] píxeles
    f_patch = np.clip(f_patch, -3, 3)
    g_patch = np.clip(g_patch, -3, 3)

    # Colocar el parche en la posición seleccionada en las matrices de desplazamiento
    disp_x[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size] = f_patch
    disp_y[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size] = g_patch

    return disp_x, disp_y

def apply_deformation(Image_Ref_noisy, Image_Ref_interpol_noisy, Xp_subset, Yp_subset, disp_x, disp_y):
    """
    Aplica la deformación directamente a la imagen de referencia con ruido.
    """
    # Crear la imagen deformada como una copia de la imagen de referencia con ruido
    Image_BD_noisy = Image_Ref_noisy.copy()

    # Encontrar dónde el desplazamiento es distinto de cero (el parche deformado)
    non_zero_mask = (disp_x != 0) | (disp_y != 0)

    # Obtener los índices de los elementos no cero
    indices = np.where(non_zero_mask)

    # Aplicar los desplazamientos al parche
    x_new = Xp_subset[indices] + disp_x[indices]
    y_new = Yp_subset[indices] + disp_y[indices]

    # Ajustar las coordenadas por el padding
    x_new_interp = x_new + 2
    y_new_interp = y_new + 2

    # Interpolación bicúbica solo en el parche sobre la imagen con ruido
    interpolated_values = map_coordinates(
        Image_Ref_interpol_noisy,
        [y_new_interp, x_new_interp],
        order=3,
        mode='reflect'
    )

    # Asignar los valores interpolados a la imagen deformada
    Image_BD_noisy[indices] = interpolated_values

    return Image_BD_noisy

def add_poisson_noise(Image, scaling=1.0):
    """
    Agrega ruido de Poisson a la imagen.
    """
    # Escalar la imagen para controlar la intensidad del ruido
    Image_scaled = Image * scaling

    # Crear una máscara donde lam >= 0 y no es NaN
    valid_mask = (Image_scaled >= 0) & (~np.isnan(Image_scaled))

    # Inicializar Image_noisy con los valores originales
    Image_noisy = Image.copy()

    # Aplicar Poisson noise solo donde valid_mask es True
    Image_noisy[valid_mask] = np.random.poisson(Image_scaled[valid_mask]).astype(np.float64) / scaling

    # Asegurar que los valores estén en [0, 255]
    Image_noisy = np.clip(Image_noisy, 0, 255)

    return Image_noisy

def apply_attenuation(Image, attenuation_mask):
    """
    Aplica atenuación aleatoria entre 50% y 100% a las regiones especificadas de la imagen.
    """
    # Aplicar atenuación
    attenuation = 0.5 + 0.5 * random.random()  # Entre 0.5 y 1
    Image[attenuation_mask] *= attenuation

    # Asegurar que los valores estén en [0, 255]
    Image = np.clip(Image, 0, 255)

    return Image

def save_data(data_dir, img_idx, l, Image_Ref_noisy, Image_BD_noisy, disp_x, disp_y):
    """
    Guarda las imágenes y mapas de desplazamiento en archivos CSV.
    Retorna la lista de nombres de archivo.
    """
    name_ref = f'Ref{img_idx:03d}_{l:03d}.csv'
    name_def = f'Def{img_idx:03d}_{l:03d}.csv'
    name_dispx = f'Dispx{img_idx:03d}_{l:03d}.csv'
    name_dispy = f'Dispy{img_idx:03d}_{l:03d}.csv'

    # Guardar los datos
    np.savetxt(os.path.join(data_dir, name_ref), Image_Ref_noisy, delimiter=",", fmt='%.6f')
    np.savetxt(os.path.join(data_dir, name_def), Image_BD_noisy, delimiter=",", fmt='%.6f')
    np.savetxt(os.path.join(data_dir, name_dispx), disp_x, delimiter=",", fmt='%.6f')
    np.savetxt(os.path.join(data_dir, name_dispy), disp_y, delimiter=",", fmt='%.6f')

    # Retornar la lista de nombres de archivo
    return [name_ref, name_def, name_dispx, name_dispy]

def process_image(args):
    """
    Procesa una única imagen para generar augmentaciones de entrenamiento y prueba.
    """
    (image_path, train_data_dir, test_data_dir, n_train, n_test,
     SubsetSize, patch_sizes, train_annotations, test_annotations) = args
    # Ajusta esta parte según cómo estén nombrados tus archivos.
    # Ejemplo: si tu archivo se llama "I_ref_step_0.tif", extrae el número final:
    img_stem = Path(image_path).stem  # p.ej. "I_ref_step_0"
    # Suponiendo que el número esté siempre tras el último '_':
    try:
        img_idx = int(img_stem.split('_')[-1])
    except:
        # Si no logras parsear, asigna un idx = 0 o algo por defecto
        img_idx = 0

    try:
        # Leer imagen de referencia
        Image_Ref = io.imread(image_path)
        # Convertir a escala de grises si es necesario
        if Image_Ref.ndim == 3:
            Image_Ref = rgb2gray(Image_Ref)
        # Asegurarse de que la imagen es de tipo float64
        Image_Ref = Image_Ref.astype(np.float64)

        # Comprobar los valores de la imagen
        print(f"Procesando imagen {img_idx}, forma: {Image_Ref.shape}, min: {Image_Ref.min()}, max: {Image_Ref.max()}")

        # Verificar el tamaño de la imagen
        if Image_Ref.shape[0] != SubsetSize or Image_Ref.shape[1] != SubsetSize:
            print(f"Advertencia: La imagen {img_idx} tiene un tamaño diferente a {SubsetSize}x{SubsetSize}")
            from skimage.transform import resize
            Image_Ref = resize(Image_Ref, (SubsetSize, SubsetSize), mode='reflect', anti_aliasing=True) * 255
            Image_Ref = Image_Ref.astype(np.float64)

        # Agregar ruido de Poisson a la imagen de referencia
        scaling_ref = 1.0
        Image_Ref_noisy = add_poisson_noise(Image_Ref, scaling=scaling_ref)

        # Verificar que no haya NaNs ni valores negativos
        assert not np.isnan(Image_Ref_noisy).any(), "Image_Ref_noisy contiene NaNs después de agregar ruido de Poisson"
        assert (Image_Ref_noisy >= 0).all(), "Image_Ref_noisy contiene valores negativos después de agregar ruido de Poisson"

        # Preparar la imagen con padding para la interpolación bicúbica (usando la imagen con ruido)
        Image_Ref_interpol_noisy = np.pad(Image_Ref_noisy, pad_width=2, mode='reflect')

        # Crear las grillas de coordenadas para la interpolación
        Xp_subset, Yp_subset = np.meshgrid(np.arange(SubsetSize), np.arange(SubsetSize))

        # Generación de datos de entrenamiento
        for l in range(1, n_train + 1):
            disp_x, disp_y = generate_single_patch_displacement(patch_sizes, SubsetSize)
            assert np.all((disp_x >= -4) & (disp_x <= 4)), "Desplazamiento_x fuera de rango"
            assert np.all((disp_y >= -4) & (disp_y <= 4)), "Desplazamiento_y fuera de rango"

            # Aplicar deformación directamente a la imagen con ruido
            Image_BD_noisy = apply_deformation(
                Image_Ref_noisy, Image_Ref_interpol_noisy, Xp_subset, Yp_subset, disp_x, disp_y
            )

            # Agregar ruido de Poisson a la imagen deformada
            scaling_bd = 1.0
            Image_BD_noisy = add_poisson_noise(Image_BD_noisy, scaling=scaling_bd)

            # Verificar que Image_BD_noisy no contenga NaNs o valores negativos
            assert not np.isnan(Image_BD_noisy).any(), "Image_BD_noisy contiene NaNs después de agregar ruido de Poisson"
            assert (Image_BD_noisy >= 0).all(), "Image_BD_noisy contiene valores negativos después de agregar ruido de Poisson"

            # Generar máscara de atenuación
            attenuation_mask = (disp_x != 0) | (disp_y != 0)
            Image_BD_noisy = apply_attenuation(Image_BD_noisy, attenuation_mask)

            # Guardar datos
            filenames = save_data(train_data_dir, img_idx, l, Image_Ref_noisy, Image_BD_noisy, disp_x, disp_y)
            train_annotations.append(filenames)

        # Generación de datos de prueba
        for l in range(1, n_test + 1):
            disp_x, disp_y = generate_single_patch_displacement(patch_sizes, SubsetSize)
            assert np.all((disp_x >= -4) & (disp_x <= 4)), "Desplazamiento_x fuera de rango"
            assert np.all((disp_y >= -4) & (disp_y <= 4)), "Desplazamiento_y fuera de rango"

            Image_BD_noisy = apply_deformation(
                Image_Ref_noisy, Image_Ref_interpol_noisy, Xp_subset, Yp_subset, disp_x, disp_y
            )
            Image_BD_noisy = add_poisson_noise(Image_BD_noisy, scaling=1.0)

            assert not np.isnan(Image_BD_noisy).any(), "Image_BD_noisy contiene NaNs después de agregar ruido de Poisson"
            assert (Image_BD_noisy >= 0).all(), "Image_BD_noisy contiene valores negativos después de agregar ruido de Poisson"

            attenuation_mask = (disp_x != 0) | (disp_y != 0)
            Image_BD_noisy = apply_attenuation(Image_BD_noisy, attenuation_mask)

            filenames = save_data(test_data_dir, img_idx, l, Image_Ref_noisy, Image_BD_noisy, disp_x, disp_y)
            test_annotations.append(filenames)

    except Exception as e:
        print(f"Error procesando la imagen {img_idx}: {e}")

def main():
    from pathlib import Path

    print("Directorio actual:", Path.cwd())

    # En lugar de usar base_dir = Path.cwd(), apuntemos directo a la carpeta Data_augmentation
    # O, si prefieres, puedes usar base_dir = Path.cwd() y luego concatenar 'Data_augmentation'.
    
    base_dir = Path.cwd()  # Directorio actual (donde estás ejecutando el script)
    # dataset_dir lo puedes dejar como quieras, pero si quieres que se cree dentro de Data_augmentation:
    # dataset_dir = base_dir / 'Data_augmentation' / 'Dataset'
    
    # O si prefieres seguir la estructura que tenías (carpeta hermana a Data_augmentation):
    dataset_dir = base_dir.parent / 'Dataset'
    
    train_data_dir = dataset_dir / 'Train_Data'
    test_data_dir = dataset_dir / 'Test_Data'
    
    # Creamos las carpetas de salida
    train_data_dir.mkdir(parents=True, exist_ok=True)
    test_data_dir.mkdir(parents=True, exist_ok=True)

    # Aquí ajustamos para que busque la carpeta Train_References_1 dentro de Data_augmentation
    train_references_dir = base_dir / 'Data_augmentation' / 'Train_References_1'
    
    print("Buscando en:", train_references_dir)
    print("¿Existe el directorio?", train_references_dir.exists())

    # Ver todos los archivos sin filtro
    all_files = list(train_references_dir.glob('*'))
    print("Archivos en la carpeta (sin filtro):", all_files)

    # Ahora filtramos solo los .tif
    image_files = list(train_references_dir.glob('*.tif'))
    print("Archivos con extensión .tif:", image_files)

    num_images = len(image_files)
    print("Número de imágenes encontradas:", num_images)

    if num_images == 0:
        print("No se encontraron imágenes en la carpeta Train_References_1")
        return

    # Definir parámetros
    n_train = 80              # Número de variaciones por imagen (entrenamiento)
    n_test = 20               # Número de variaciones por imagen (prueba)
    SubsetSize = 256          # Tamaño de las imágenes
    patch_sizes = [4, 8, 16, 32, 64, 128]

    # Usar Manager para compartir las listas entre procesos
    with Manager() as manager:
        train_annotations = manager.list()
        test_annotations = manager.list()

        # Preparar argumentos para el procesamiento paralelo
        args_list = [
            (str(image_path), str(train_data_dir), str(test_data_dir),
             n_train, n_test, SubsetSize, patch_sizes, train_annotations, test_annotations)
            for image_path in image_files
        ]

        # Procesamiento en paralelo
        with Pool(processes=cpu_count()) as pool:
            pool.map(process_image, args_list)

        # Ordenar anotaciones y guardarlas
        train_annotations_sorted = sorted(train_annotations)
        test_annotations_sorted = sorted(test_annotations)

        train_annotations_file = dataset_dir / 'Train_annotations.csv'
        with open(train_annotations_file, 'w') as f:
            for filenames in train_annotations_sorted:
                f.write(','.join(filenames) + '\n')

        test_annotations_file = dataset_dir / 'Test_annotations.csv'
        with open(test_annotations_file, 'w') as f:
            for filenames in test_annotations_sorted:
                f.write(','.join(filenames) + '\n')

    print("Augmentación de datos completada.")

if __name__ == "__main__":
    main()
