import laspy
import os

# print('Running Merge LAS')

#This is the las file to append to.  DO NOT STORE THIS FILE IN THE SAME DIRECTORY AS BELOW...
# out_las = 'D:/Train Model SBY for AHY/Model M-DGCNN RGB + Intensity/experiment_3class/log/sem_seg/sby/visual/2_F12 M350 Classified - Test - all_points.las'
#this is a directory of las files
# inDir = 'D:/Train Model SBY for AHY/Model M-DGCNN RGB + Intensity/experiment_3class/log/sem_seg/sby/visual/2_F12 M350 Classified - Test/'    

def append_to_las(in_las, out_las):
    with laspy.open(out_las, mode='a') as outlas:
        with laspy.open(in_las) as inlas:
            for points in inlas.chunk_iterator(2_000_000):
                outlas.append_points(points)

'''
for (dirpath, dirnames, filenames) in os.walk(inDir):
    for inFile in filenames:
        if inFile.endswith('.las'):
            in_las = os.path.join(dirpath, inFile)
            append_to_las(in_las, out_las)
    
print('Finished without errors - merge_LAS.py')
'''