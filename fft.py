from asyncio.windows_events import NULL

import argparse
import math 
import cv2
import time
import statistics

import numpy as np
import matplotlib.pyplot as mathplt
import matplotlib.colors as colors


# Apply naive DFT formula
def naive_1D_DFT(array):
    array = np.asarray(array, dtype=complex)
    N = len(array)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.matmul(M, array)


# Apply naive IDFT formula
def naive_1D_IDFT(array):
    array = np.asarray(array, dtype=complex)
    N = len(array)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    return np.matmul(M/N, array)


# Apply the Cooley-Tukey FFT formula
def fast_1D_DFT(array):
    array = np.asarray(array, dtype=complex)
    N = len(array)
    
    # If it's not a power of 2, we can't use divide and conquer approach
    if(N % 2 == 0):
        # Set smallest size sub-array for divide and conquer approach to 8.
        # Once we have a sub-array of size 8, we resort to using the naive DFT approach.
        if(N <= 8):
            return naive_1D_DFT(array)
        
        # If not, we separate the array into two: the even terms and odd terms.
        # Just as described by the Cooley-Tukey FFT, we recursively divide the array
        # into smaller even and odd terms till they are of size 8.
        evenTermDFTs = fast_1D_DFT(array[::2])
        oddTermDFTs = fast_1D_DFT(array[1::2])
        n = np.arange(N)
        
        # Once reaching that size 8 state, we will have all of the even and odd terms DFTs.
        # We calculate the exponential factor and apply the formula.
        exponential = np.exp(-2j * np.pi * n / N)
        evenTermDFTs = np.insert(evenTermDFTs, len(evenTermDFTs), evenTermDFTs)
        oddTermDFTs = np.insert(oddTermDFTs, len(oddTermDFTs), oddTermDFTs)
        return evenTermDFTs + exponential * oddTermDFTs
    else:
        print("Error!   Array must have a size of a power of 2")
        exit(1)
    
# Apply the Cooley-Tukey FFT formula but now the inverse
def fast_1D_IDFT(array):
    array = np.asarray(array, dtype=complex)
    N = len(array)
    
     # If it's not a power of 2, we can't use divide and conquer approach
    if(N % 2 == 0):
        # S et smallest size sub-array for divide and conquer approach to 8.
        # Once we have a sub-array of size 8, we resort to using the naive IDFT approach.
        if(N <= 8):
            return naive_1D_IDFT(array)*N
        
        # If not, we separate the array into two: the even terms and odd terms.
        # Just as described by the Cooley-Tukey FFT, we recursively divide the array
        # into smaller even and odd terms till they are of size 8.
        # Give NULL because we don't want to divide by N for these calls, only the first one
        evenTermDFTs = fast_1D_IDFT(array[::2])   
        oddTermDFTs = fast_1D_IDFT(array[1::2])
        n = np.arange(N)
        
        # Once reaching that size 8 state, we will have all of the even and odd terms IDFTs.
        # We calculate the exponential factor and apply the formula.
        exponential = np.exp(2j * np.pi * n / N)
        evenTermDFTs = np.insert(evenTermDFTs, len(evenTermDFTs), evenTermDFTs)
        oddTermDFTs = np.insert(oddTermDFTs, len(oddTermDFTs), oddTermDFTs)
        

        # NOTE: at the end of the first ever call to this method, we want to divide this return
        # value by N. So every time this method is called, make sure the returned output is divided by N.
        return evenTermDFTs + exponential * oddTermDFTs
    else:
        print("Error!   Array must have a size of a power of 2")
        exit(1)


# Apply the 2D Formula with FFT
def fast_2D_DFT(array):
    ans = np.zeros(array.shape, dtype=complex)

    for i in range(array.shape[1]):
        ans[:, i] = fast_1D_DFT(array[:, i])
    for i in range(array.shape[0]):
        ans[i, :] = fast_1D_DFT(ans[i, :])
    return ans


# Apply the 2D Formula with Naive DFT
def naive_2D_DFT(array):
    ans = np.zeros(array.shape, dtype=complex)

    for i in range(array.shape[1]):
        ans[:, i] = naive_1D_DFT(array[:, i])
    for i in range(array.shape[0]):
        ans[i, :] = naive_1D_DFT(ans[i, :])
    return ans


# Apply the 2D Formula with IFFT
def fast_2D_IDFT(array):
    ans = np.zeros(array.shape, dtype=complex)

    # As mentioned in the Note of the fast_1D_IDTF method, make sure that the return
    # output is divided by N to respect the IFFT formula.
    for i in range(array.shape[0]):
        N = len(array[i, :])
        ans[i, :] = fast_1D_IDFT(array[i, :])/N
    for i in range(array.shape[1]):
        N = len(ans[:, i])
        ans[:, i] = fast_1D_IDFT(ans[:, i])/N
    return ans


# Apply the 2D Formula with Naive IDFT
def naive_2D_IDFT(array):
    ans = np.zeros(array.shape, dtype=complex)

    for i in range(array.shape[0]):
        ans[i, :] = naive_1D_IDFT(array[i, :])
    for i in range(array.shape[1]):
        ans[:, i] = naive_1D_IDFT(ans[:, i])
    return ans


def runTests():
    d1 = np.random.random(128)
    d2 = np.random.rand(16, 16)

    if not np.allclose(naive_1D_DFT(d1), np.fft.fft(d1)):
        print("DTF Naive Error")
    
    if not np.allclose(naive_1D_IDFT(d1), np.fft.ifft(d1)):
        print("IDTF Naive Error")
        
    if not np.allclose(fast_1D_DFT(d1), np.fft.fft(d1)):
        print("FFT Error")
    
    # As mentioned in the Note of the fast_1D_IDTF method, make sure that the return
    # output is divided by N to respect the IFFT formula.
    N = len(d1)
    if not np.allclose(fast_1D_IDFT(d1)/N, np.fft.ifft(d1)):
        print("IFFT Error")
        
    if not np.allclose(naive_2D_DFT(d2), np.fft.fft2(d2)):
        print("2D Naive DFT Error")
        
    if not np.allclose(naive_2D_IDFT(d2), np.fft.ifft2(d2)):
        print("2D Naive IDFT Error")
        
    if not np.allclose(fast_2D_DFT(d2), np.fft.fft2(d2)):
        print("2D FFT Error")
        
    if not np.allclose(fast_2D_IDFT(d2), np.fft.ifft2(d2)):
        print("2D IFFT Error")


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', action='store', dest='mode', type=int, default=1)
    parser.add_argument('-i', action='store', dest='image', type=str, default='moonlanding.png')
    args= parser.parse_args()
    
    mode = args.mode
    image = args.image
    
    runTests()
    
    # Take input image
    img = mathplt.imread(image).astype(float)

    # Padding image to desired size
    width = int(pow(2, math.ceil(math.log2(img.shape[0]))))
    height = int(pow(2, math.ceil(math.log2(img.shape[1]))))
    img = cv2.resize(img, (height, width))
    
    # Display image in FFT form
    if(mode == 1):
        
        # Run 2D FFT
        FFT_of_img = fast_2D_DFT(img)

        # Show graph
        fig, ax = mathplt.subplots(1, 2)
        ax[0].imshow(img, mathplt.cm.gray)
        ax[0].set_title('Original')
        ax[1].imshow(np.abs(FFT_of_img), norm=colors.LogNorm())
        ax[1].set_title('Fourier Transform With LogNorm Applied')
        fig.suptitle('Mode 1')
        mathplt.show()
        
    # Denoise image
    elif(mode == 2):
        
        # Run 2D FFT
        FFT_of_img = fast_2D_DFT(img)
        
        # Apply Filter
        denoiseFactor = 0.075
        X, Y = FFT_of_img.shape
        xPixels = int(denoiseFactor * X)
        yPixels = int(denoiseFactor * Y)
        print("The Denoising is keeping", denoiseFactor*100, "% of non-zero values")
        print("Represents (", xPixels, ",", yPixels,") from total of (",X,",",Y,")")
        
        
        FFT_of_img[ xPixels : int((1 - denoiseFactor) * X)] = 0
        FFT_of_img[:, yPixels : int((1 - denoiseFactor) * Y)] = 0
    
        # Run 2D IFFT
        IFFT_of_img = fast_2D_IDFT(FFT_of_img).real
        
        # Show graph
        fig, ax = mathplt.subplots(1, 2)
        ax[0].imshow(img, mathplt.cm.gray)
        ax[0].set_title('Original')
        ax[1].imshow(IFFT_of_img[:img.shape[0], :img.shape[1]], mathplt.cm.gray)
        ax[1].set_title('Denoise Applied Image ({}%)'.format(denoiseFactor*100))
        fig.suptitle('Mode 2')
        mathplt.show()
    
    # Compress image    
    elif(mode == 3):
        
        # Run 2D FFT
        FFT_of_img = fast_2D_DFT(img)
        # Original size of the image
        realSize = img.shape[0]*img.shape[1]
        
        # The six different compression levels applied to the image
        compressionLevels = [0, 18, 48, 65, 85, 99.9]
        # Will hold all of the compressed images after applying the filter on the FFT and doing the IFFT
        compressedImgs = []
        
        # Iterate through all the compression levels, apply the compression filter and run the IFFT to
        # obtain the compressed image. Add all compressed images to the array.
        for level in compressionLevels:
            print("The Compression of", level, "% is keeping", int(realSize * (1-(level/100))), " out of", realSize, "non-zero values")
            filter = np.percentile(abs(FFT_of_img), level)
            FFT_of_img[abs(FFT_of_img) < filter] = 0
            compressedImg = fast_2D_IDFT(FFT_of_img)
            compressedImgs.append(compressedImg)
        
        # Show graph
        fig, ax = mathplt.subplots(2, 3, figsize=(10, 8))
        axis = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
        
        for i in range(6):
            a, b = axis[i]
            ax[a,b].imshow(np.real(compressedImgs[i])[:img.shape[0], :img.shape[1]], mathplt.cm.gray)
            ax[a,b].set_title('Compression Level {}%'.format(compressionLevels[i]))
        fig.suptitle('Mode 3')
        mathplt.show()
        
     # Plot Run Times    
    elif(mode == 4):   

        mathplt.figure(figsize=(12, 5))
        mathplt.title('Fourier Transform Runtime Analysis')
        mathplt.xlabel('Problem Size')
        mathplt.ylabel('Average Runtime (sec)')


        y_slowValues = []       # Will hold the naive DFT average runtimes
        y_fastValues = []       # Will hold the FFT average runtimes
        naiveError = []         # Will hold the naive DFT standard deviations
        fastError = []          # Will hold the FFT standard deviations
        arraySizes = [2**5, 2**6, 2**7, 2**8]   # Sizes of 2D array to be created

        # Iterate through all the 2D array sizes previously defined. Run 10 tries
        # for both the naive and fast algorithms for each array size
        # to get an average of the runtimes.
        for arraySize in arraySizes:
            array = np.random.random((arraySize, arraySize))
            naiveValues = []
            fastValues = []
            for i in range(1, 10):
                # Naive DFT Time
                start = time.time()
                naive_2D_DFT(array)
                end = time.time()
                naiveValues.append(end - start)

                # FFT Time
                start = time.time()
                fast_2D_DFT(array)
                end = time.time()
                fastValues.append(end - start)

            print("Runtime values for 2D array of size", arraySize)
            print()
            
            # For each array size, calculate and print the mean and variance
            # for both algorithms using the 10 trial values.
            # Also calculate the error.
            print("Naive DFT Values:")
            averageNaiveTime = sum(naiveValues) / 10
            y_slowValues.append(averageNaiveTime)
            naiveError.append(statistics.stdev(naiveValues) * 2)
            
            print("Naive DFT Mean: ", np.mean(naiveValues))
            print("Naive DFT Variance: ", np.var(naiveValues))
            print()
            
            print("FFT Values:")
            averageFastTime = sum(fastValues) / 10
            y_fastValues.append(averageFastTime)
            fastError.append(statistics.stdev(fastValues) * 2)
            
            print("FFT Mean: ", np.mean(fastValues))
            print("FFT Variance: ", np.var(fastValues))
            print()

        # Show graph
        mathplt.errorbar(x=['32x32', '64x64', '128x128', '256x256'], y=y_slowValues, yerr=naiveError, label='Naive DFT')
        mathplt.errorbar(x=['32x32', '64x64', '128x128', '256x256'], y=y_fastValues, yerr=fastError, label='FFT')
        mathplt.legend(loc='upper left')
        mathplt.show()
    
    return 0










if __name__ == '__main__':
    main()