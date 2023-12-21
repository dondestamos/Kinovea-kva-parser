import numpy as np
import pandas as pd
import os
from tkinter import filedialog
import re
import linecache
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm # Progress bar for the long cycle-parsing loop
import time # Progress bar for the long cycle-parsing loop
import sys



StrOut = ('This script will parse a .kva (Kinovea Annotation) file, extracting marker locations in each frame.\n'
        'Two reasons for using it instead using the "Linear Kinematics" (LK) module in Kinovea:\n'
        '1. Only one calibration at a time is supported in Kinovea. As we adopted 2- or 3-camera marker recording at once, a few video sourced composed together and tracked at once for faster file handling. Therefore we need to calibrate each video source separately. The script does that, while being backward-compatible with the global calibration.\n'
        '2. The LK module performs filtering at multiple cutoff frequencies to determine the optimal one. While versatile, this approach hangs up the app when long videos are tracked and often causes it to crash. The script displays a progress bar and is robust in long files with multiple markers. It still does filtering at user-specified cutoff frequency.\n'
        'After the .kva is parsed, you will be asked in this CMD session to specify lengths of the line(s) drawn for calibration, map each line to the markers which will be calibrated by it, adjust filtering cutoff frequency, and verify the strict output order of the markers. \n'
        'The data will be saved as a CSV file in the same folder where the video file was.\n\n'
        'Warning: most of the input parsers here are not robust to format errors. Pay close attention to required format, e.g. with/out spaces or with/out commas. Use millimiters when calibrating.')
print(StrOut)
print('\n')
input('Press Enter to continue to selecting a .kva file')
print('\n')

# Automatic mapping of calibration lines and markers is possible when they have consistent naming with underscores, e.g.:
# ProximalMaaa_1, DistalBoo_1, LineJaaa_1

# For labeling in Kinovea:
# 1. Create distal marker first, then the corresponding proximal marker. Order matters
# 2. Name them with underscore and number, Prox_1 and Dist_1, Proximal_2 and Distal_2 etc.
# 3. Create a line on an object of a known dimension (use one of the markers. Pushpins are 3/8" = 9.5 mm and reflective markers overall diam = 10 mm). 
# 4. Name that line with the underscore and the same number, as the closest pair of markers, e.g. Line_1
# 5. If several video sources are composed into one, create a line in each, Line_1, Line_2 etc.
# 6. Do not calibrate the lines in Kinovea, unless you want all the markers be calibrated by this single line (prone to errors and confusions)


CurFolder = current_directory = os.path.dirname(os.path.abspath(__file__))
# Prompt user for the KVA file
FilePath = filedialog.askopenfilename(title="Select the KVA File to process", filetypes=[("Kinovea Annotation", "*.kva"), ("All Files", "*.*")], initialdir=CurFolder)
#FileName = "Test_60HL_Test"

FileName = FilePath.split('/')[-1]
print(FileName)

# Grab basic properties of the video file
with open(FilePath,'r') as file:
    lines = file.readlines()
    file.seek(0)
    #flag_pauseUser = 0
    ImSizeLine = [line.strip() for line in file if "<ImageSize>" in line]
    ImX = int(re.split(r'[<;>]',ImSizeLine[0])[2])
    ImY = int(re.split(r'[<;>]',ImSizeLine[0])[3])
    file.seek(0)
    FPSLine = [line.strip() for line in file if "<CaptureFramerate>" in line]
    FPS = re.split(r'[<>]',FPSLine[0])[2]

# Default values
LineDataAll = 'No lines present'
MarkerNames = []
MarkerData = []
LineNames = []
LineData = []
LineCal = []


# Parse the kva file detecting the values corresponding to (1) Marker Coordinates, (2) Line-drawings, (3) Calibrated Lines
with open(FilePath,'r') as file:
    lines = file.readlines()
    file.seek(0)
    flag_activeCal = 0
    flag_activeMark = 0
    flag_activeLine = 0
    

    for iline, line in tqdm(enumerate(file), total=len(lines), desc='Parsing',unit='lines',ncols=75):
        # Detect markers
        flag_activeMark = (flag_activeMark | ("<Track id=" in line)) & (not("/TrackPointList" in line))
        # Detect marker name
        if "<Track id=" in line:
            MarkName = line.split('name="')[1].split('">')[0]
            MarkerNames.append(MarkName)
            imark = len(MarkerNames)
            XName = MarkName + '_X'
            YName = MarkName + '_Y'
        # Wrap up marker's data
        if "/TrackPointList" in line:
            # Create a dataframe table if the first marker
            if imark == 1:
                MarkerDataAll = pd.DataFrame(MarkerData)
                MarkerData = [] # ready to be filled in by the next marker's data
            else: # merge with the previous marker's data
                #
                # print(f'DEBUG. Marker {imark} {MarkerNames[imark-1]}')
                MarkerData = pd.DataFrame(MarkerData)
                MarkerData.loc[:,'Time'] = MarkerData.loc[:,'Time'].round(2)
                MarkerData.set_index('Time',inplace=True)
                MarkerDataAll.set_index('Time',inplace=True)
                MarkerDataAll = MarkerDataAll.join(MarkerData, how='left')
                MarkerDataAll = MarkerDataAll.rename(columns={'index': 'Time'})
                MarkerDataAll.reset_index(inplace=True)
                MarkerData = []

                # Alternative
                # merged_df = pd.merge(df1, df0, on='Time', how='left')

        # Detect drawn lines, if any
        flag_activeLine = (flag_activeLine | ("Line id=" in line)) & (not ("/Line" in line))
        if "Line id=" in line:
            LineName = line.split('name="')[1].split('">')[0]
            LineNames.append(LineName) 
            
        # Wrap up lines
        if "</Drawings>" in line:
            LineDataAll = pd.DataFrame(LineData)

        # Detect calibration, if any
        flag_activeCal = (flag_activeCal | ("<Calibration>" in line)) & (not ("</Calibration>" in line))
        if flag_activeCal & ("Length" in line):
            LUnit = float(line.split('<Length>')[1].split('</Length')[0])
        if flag_activeCal & ("<A>" in line):
            CalX1 = float(line.split('<A>')[1].split(';')[0])
            CalY1 = float(line.split(';')[1].split('</A')[0])
        if flag_activeCal & ("<B>" in line):
            CalX2 = float(line.split('<B>')[1].split(';')[0])
            CalY2 = float(line.split(';')[1].split('</B')[0])
            Lpx = np.sqrt((CalX1 - CalX2)**2 + (CalY1 - CalY2)**2)
        if flag_activeCal & ("Unit Abbreviation" in line):
            Units = line.split('>')[1].split('</')[0]
            CalPx2Unit = round(LUnit / Lpx,4)
            if CalPx2Unit == 1:
                CalibrationInfo = f'No pre-calibrated lines available, coordinates are in Pixels'
            else:
                CalibrationInfo = f'A pre-calibrated line is present, {LUnit} {Units} long, calibration coefficient {CalPx2Unit} {Units}/Pixel'

        #print(f'DEBUG. Line {iline}: {line}')

        if not (flag_activeMark | flag_activeCal | flag_activeLine):
            continue
        
        # Detect marker coordinates
        if "TrackPoint UserX" in line:
            CoordLine = line
            X = float(line.split('"')[1])
            Y = float(line.split('"')[5])
            
            Tstr = line.split('UserTime="')[1].split('"')[0]
            Ncolon = len([ch for ch in Tstr if ch==":"]) 
            if Ncolon == 0: # Less than 1:00.00(1 )
                T = float(Tstr)
            elif Ncolon == 1: # minuites and seconds
                Tts = pd.to_datetime(Tstr,format="%M:%S.%f")
                T = Tts.minute * 60 + Tts.second + Tts.microsecond / 1000000
            else: 
                Tts = pd.to_datetime(Tstr,format="%H%M:%S.%f")
                T = Tts.hour * 3600 + Tts.minute * 60 + Tts.second + Tts.microsecond / 1000000

            New_Row = {'Time':T, XName:X, YName:Y}
            MarkerData.append(New_Row)
        
        # Detect Line coordinates
        if "Start" in line:
            LineX1 = float(line.split('<Start>')[1].split(';')[0])
            LineY1 = float(line.split(';')[1].split('</Start')[0])
        if "End" in line:
            LineX2 = float(line.split('<End>')[1].split(';')[0])
            LineY2 = float(line.split(';')[1].split('</End')[0])
            LineL = np.sqrt((LineX1 - LineX2)**2 + (LineY1 - LineY2)**2)
            LineC_X = (LineX1 + LineX2) / 2
            LineC_Y = (LineY1 + LineY2) / 2
            New_Row = {'Name':LineName, 'Pixel L':LineL, 'Center_X':LineC_X, 'Center_Y':LineC_Y}
            LineData.append(New_Row)

# For markers, the default (if no calibration) origin is frame center, and axes surprisingly are classical X rightwards, Y upwards.
# For lines, the default (if no calibration) origin is upper left corner, and axes are X rightwards, but Y downwards.
# Shoutout to Kinovea with transparent implementation!!!
# For more intuitive process, translate origin to bottom-left corner and transform coordinates into normalized by video size.
for col in MarkerDataAll.columns:
    if col == 'Time':
        continue
    if ('x' in col) | ('X' in col):
        MarkerDataAll.loc[:,col] = MarkerDataAll.loc[:,col] + ImX/2
        MarkerDataAll.loc[:,col] = MarkerDataAll.loc[:,col] / ImX
    if ('y' in col) | ('Y' in col):
        MarkerDataAll.loc[:,col] = MarkerDataAll.loc[:,col] + ImY/2
        MarkerDataAll.loc[:,col] = MarkerDataAll.loc[:,col] / ImY
for col in LineDataAll.columns:
    if (col == 'Name') or (col == 'Pixel L'):
        continue
    if ('x' in col) | ('X' in col):
        LineDataAll.loc[:,col] = LineDataAll.loc[:,col] / ImX
    if ('y' in col) | ('Y' in col):
        LineDataAll.loc[:,col] = - LineDataAll.loc[:,col] + ImY
        LineDataAll.loc[:,col] = LineDataAll.loc[:,col] / ImY


# Prepare brief information about detected markers and lines - in case manual calibration will be used
MarkC = []
for iimark in range(0,imark):
    Temp = MarkerDataAll.mean()[(1 + iimark*2):(3 + iimark*2)]
    MarkC.append(f'{MarkerNames[iimark]} at ({round(Temp.iloc[0],2)},{round(Temp.iloc[1],2)})')

LineC = []
for iiline in range(0,len(LineDataAll)):
    LineC.append(f'{LineDataAll.iloc[iiline,0]} at ({round(LineDataAll.iloc[iiline,2],2)},{round(LineDataAll.iloc[iiline,3],2)}), L={round(LineDataAll.iloc[iiline,1],1)} px')

# print(CalibrationInfo)

if not (CalPx2Unit == 1):
    print(f'Calibration from Kinovea detected with factor {CalPx2Unit} {Units}/px. Origin and reference frame orientation are arbitrary (kudos to Kinovea), but linear scalings are hopefully preserved.')
    if imark <=2:
        print(f'{imark} markers detected: {MarkerNames}. Their positions will be automatically calibrated.')
    else:
        inStr = input(f'{imark} markers detected: {MarkerNames}. There might be different video sources with different scales composed in one. It is recommended' 
              'that you remove the calibrated line in Kinovea and instead add a non-calibrated line to a known-size object in each video source. Continue calibration of all markers by the same factor (Y) or abort (anykey)?')
        if (inStr == "Y") or (inStr == "y"):
            print('Markers positions will be automatically calibrated.')
        else:
            print('Aborting.')
            sys.exit()
else:
    print(f'Normalized coordinates are used until calibration, origin at bottom-left, XY like in school')
    print(f'{imark} markers detected and are, on average, at: {MarkC}')
    print(f'{len(LineDataAll)} lines detected and are centered at: {LineC}')
print('\n')



# Calibrate. Either use Kinovea calibration (if it can be safely applied to all the markers), or ensure that 
# for N pairs of markers there were N lines in Kinovea *without* calibrated length, i.e. in original pixels.
# Proceeding in somewhat interactive form, calibrate. If the user was smart when labeling stuff in Kinovea, they can
# just keep pressing Enter with empty inputs.




if CalPx2Unit == 1:
    LineDataAll.loc[:,'Units'] = None
    LineDataAll['Units'] = LineDataAll['Units'].astype(str)
    if 2 * len(LineDataAll) == imark:
        print('Enter to continue if the true length is exactly 9.5 mm (3/8", the colored pushpin) or specify true lengths of each line in MM, and WITH units and with space (e.g. 10 mm as the overall diameter of a reflective marker):')
        for iline in range(0,len(LineDataAll)):
            flag_try = 1
            while flag_try:
                try:
                    strIn = input(f'{LineDataAll.loc[iline,"Name"]} at ({round(LineDataAll.loc[iline,"Center_X"],1)},{round(LineDataAll.loc[iline,"Center_Y"],1)}): ')
                    if len(strIn) == 0:
                        LineDataAll.loc[iline,"True L"] = 9.5
                        LineDataAll.loc[iline,"Units"] = 'mm'
                        Units = 'mm'
                    else:
                        LineDataAll.loc[iline,"True L"] = float(strIn.strip().split(' ')[0])
                        LineDataAll.loc[iline,"Units"] = strIn.strip().split(' ')[1]
                        Units = LineDataAll.loc[iline,"Units"]
                    flag_try = 0
                except:
                    print('Check your input and retry.')

            LineCal.append(LineDataAll.loc[iline,"True L"] / LineDataAll.loc[iline,"Pixel L"])
        print('\n')    
        print('Now specify the names of two markers which will be calibrated with it.')
        for iline in range(0,len(LineDataAll)):
            
            if '_' in LineNames[iline]:
                iLn = LineNames[iline].split('_')[1]
            else:
                iLn = LineNames[iline].split(' ')[1]

            MatchingMark = []
            MatchingMark = [string for string in MarkerNames if iLn in string]
            if len(MatchingMark) == 2:
                strIn = input(f'Markers to be calibrated with the line {LineDataAll.loc[iline,"Name"]}. Autosuggested based on a matching number {iLn} in two marker names: {MatchingMark}. Enter to accept or input the two names comma-separated, no-space.')
                if len(strIn) == 0:
                    Ln2Mark_1 = MatchingMark[0]
                    Ln2Mark_2 = MatchingMark[1]
                else:
                    Ln2Mark_1 = strIn.split(',')[0]
                    Ln2Mark_2 = strIn.split(',')[1]
            else:
                MatchingMark = MarkerNames[(iline*2):(iline*2+2)]
                strIn = input(f'Markers to be calibrated with the line {LineDataAll.loc[iline,"Name"]}. Autosuggested based on order of lines and trackers: {MatchingMark}. Enter to accept or input the two names comma-separated, no-space.')
                if len(strIn) == 0:
                    Ln2Mark_1 = MatchingMark[0]
                    Ln2Mark_2 = MatchingMark[1]
                else:
                    Ln2Mark_1 = strIn.split(',')[0]
                    Ln2Mark_2 = strIn.split(',')[1]
            # Calibrate. 
            MatchingMark = [string for string in MarkerDataAll.columns.tolist() if ((Ln2Mark_1 in string) | (Ln2Mark_2 in string))]
            MarkerDataAll.loc[:,MatchingMark] = MarkerDataAll.loc[:,MatchingMark] * LineCal[iline]
            
    elif len(LineDataAll) == 1:
        strIn = input('A single line will be used to calibrate all the markers'' positions. Enter to continue if the length is 10 mm exactly, or specify its true length with units, with space (e.g. 10 mm):')
        if len(strIn) == 0:
            LineDataAll.loc[0,"True L"] = 10
            LineDataAll.loc[0,"Units"] = 'mm'
        else:
            LineDataAll.loc[0,"True L"] = float(strIn.strip().split(' ')[0])
            LineDataAll.loc[0,"Units"] = strIn.strip().split(' ')[1]
        LineCal.append(LineDataAll.loc[0,"True L"] / LineDataAll.loc[0,"Pixel L"])
        # Calibrate.
        MarkerDataAll.iloc[:,1:] = MarkerDataAll.iloc[:,1:] * LineCal[0]
    else:
        print('Only a single line per video or a line per each pair of markers is supported. Aborting.')
        sys.exit()
 
print('\n')
print('Checking continuity of the data, i.e. whether the user moved the Kinovea time slider too fast and skipped frames')
# Check continuity in case user scrolled too much during autotracking and missed some frames.
dts = np.diff(MarkerDataAll['Time'])
if not (len([row for row in range(0,len(dts)) if np.abs(dts[row] - np.mean(dts)) > 0.5 * np.mean(dts)]) == 0):
    print('Data entries are not time-continuous, there are jumps. Filtering might introduce artifacts. Consider re-tracking.')
else:
    print('Ok')
print('\n')


# Filter the data with the same filter (and almost the same edge-padding) as in Kinovea.
flag_try = 1
while flag_try:
    strIn = input(f'Sampling rate is {FPS} fps. The data will be filtered by zero-shift second-order Butterworth filter at cutoff frequency 5 Hz. Enter to continue or type another cutoff frequency without units and without space (e.g. 3)')
    if len(strIn) == 0:
        fcut = 5
    else:
        fcut = float(strIn)
    fcut_norm = fcut / float(FPS) * 2
    if (fcut_norm > 0) & (fcut_norm < 1):
        flag_try = 0
    else:
        print('Cutoff frequency must be a number, between 0 and a half of the sampling rate. Try again')
print('\n')

sos = butter(2, fcut_norm, btype='low', analog=False, output='sos')
MarkerDataAllF = MarkerDataAll.copy()
for col in MarkerDataAll.columns[1:].tolist():
    MarkerDataAllF.loc[:,col] = sosfiltfilt(sos, MarkerDataAll.loc[:,col], padtype='odd', padlen= min(50, int(len(MarkerDataAll) / 10) ))

# Evaluate filtering by RMSE averaged across columns?..
# (MarkerDataAll - MarkerDataAllF) / MarkerDataAll

print('\n')
# Re-order columns if necessary, then rename them for shorter notation. Max 3 pairs of markers supported.
NewNames = ['D1_X','D1_Y','P1_X','P1_Y','D2_X','D2_Y','P2_X','P2_Y','D3_X','D3_Y','P3_X','P3_Y']
NewNamesM = ['D1','P1','D2','P2','D3','P3']
OldCols = MarkerDataAllF.columns[1:].to_list()
ncols = len(MarkerDataAllF.columns[1:])
print('The markers will be renamed and will go in exactly this order D1, P1, D2, P2 etc, for compatibitility with the Analyze - 1 - merge data.py. ')
strIn = input(f'Current markers go in this order {MarkerNames}. Enter to continue if it matches the target order, or type the shorter name versions comme-separated no-space (P2,D1 etc.) repeating the current markers order, for reordering.')
if len(strIn) == 0:
    MarkerDataAllF.columns = ['Time'] + NewNames[:ncols]
    strNew = NewNames[:ncols]
else:
    strIn = strIn.split(',')
    if not(len(strIn * 2) == ncols):
        print('Number of columns to re-sort does not match the original number of columns. Aborting')
        sys.exit()
    strNew = []
    for istr in range(0,len(strIn)):
        strNew.append(strIn[istr]+'_X')
        strNew.append(strIn[istr]+'_Y')
    MarkerDataAllF.columns = ['Time'] + strNew
    MarkerDataAllF = MarkerDataAllF[['Time'] + NewNames[:len(strNew)]]
    
# Remove normalization by screen size
for col in MarkerDataAllF.columns:
    if col == 'Time':
        continue
    if ('x' in col) | ('X' in col):
        MarkerDataAllF.loc[:,col] = MarkerDataAllF.loc[:,col] * ImX
    if ('y' in col) | ('Y' in col):
        MarkerDataAllF.loc[:,col] = MarkerDataAllF.loc[:,col] * ImY

# Round the data to reduce file size with minimal space sacrifice.
MarkerDataAllF = MarkerDataAllF.round(4)
for col in LineDataAll.columns:
    if (col == "Name") | (col == "Units"):
        continue
    LineDataAll.loc[:,col] = round(LineDataAll.loc[:,col],2)

# Output the data into a CSV file.
flag_LineW = 0
with open(CurFolder + '/' + FileName + '_Tracking.csv', 'w') as OutFile:
    OutFile.write(FileName + ' Kinovea tracking processed\n')
    OutFile.write(str(FPS) + ' fps\n')
    OutFile.write(str(ImX) + ' x ' + str(ImY) + ' px\n')
    if not (CalPx2Unit == 1):
        OutFile.write(f'Automatic calibration with factor {CalPx2Unit} {Units}/px was applied in Kinovea to all the markers. Origin and axes directions are arbitrary. The marker data follow.\n')
        OutFile.write(f'Note the data have arbitrary origin and r.f. orientation. The analysis is expected to only compute Prox2Dist distance and extract 1st principal component from 4 coordinates of each pair of markers.\n')
    else:
        OutFile.write(f'Calibration was manual via {len(LineCal)} line(s), details below, line coordinates normalized (marker coordinates will be calibrated though), origin bottom-left, XY like in school.\n')
        flag_LineW = 1
    if flag_LineW:
        LineDataAll.to_csv(OutFile, index=False, header=True)
    
#with open(CurFolder + '/' + FileName + '_Tracking.csv', 'a') as OutFile:    
    OutFile.write(f'The data were filtered with zero-lag second-order Butterworth low-pass filter at cutoff frequency {fcut} Hz.\n')
    OutFile.write(f'Original marker order was {OldCols}\n.')
    OutFile.write(f'They were abbreviated as {strNew} and probably re-ordered. Verify that these abbreviations match the originals.\n')
    OutFile.write(f'Marker data are calibrated to {Units}\n.')
    MarkerDataAllF.to_csv(OutFile, index=False, header=True)      
            

print(f'Done, saved in {CurFolder}/{FileName}_Tracking.csv')



