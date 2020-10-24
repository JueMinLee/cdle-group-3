package myproject;

import myproject.Helper.Helper;
import org.apache.commons.io.FileUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.util.ArchiveUtils;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;


public class DrawingIterator {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(DrawingIterator.class);
    private static final int seed = 123;
    private static Random rng = new Random(seed);
    private static String dataDir;
    //private static String downloadLink;
    private static Path trainDir, testDir;
    private static InputSplit trainData, testData;
    private static final int nChannels = 3;
    public static final int gridWidth = 13;
    public static final int gridHeight = 13;
    public static final int yolowidth = 416;
    public static final int yoloheight = 416;
    private static ImageTransform transform;
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();


    private static RecordReaderDataSetIterator makeIterator(InputSplit split, boolean training, Path dir, int batchSize) throws Exception {

        ObjectDetectionRecordReader recordReader = new ObjectDetectionRecordReader(yoloheight, yolowidth, nChannels,
                gridHeight, gridWidth, new VocLabelProvider(dir.toString()));
        if (training && transform != null){
            recordReader.initialize(split,transform);
        }else{
            recordReader.initialize(split);
        }
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, 1, true);
        iter.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        return iter;
    }

    public static RecordReaderDataSetIterator trainIterator(int batchSize) throws Exception {
        return makeIterator(trainData, true, trainDir, batchSize);
    }

    public static RecordReaderDataSetIterator testIterator(int batchSize) throws Exception {
        return makeIterator(testData, false, testDir, batchSize);
    }

    public static void setup(int batchSizeArg, int trainPerc, ImageTransform imageTransform) throws IOException {
        transform=imageTransform;
        setup(batchSizeArg,trainPerc);
    }

    public static void setup(int batchSizeArg, int trainPerc) throws IOException {
        /*dataDir = Paths.get(
                System.getProperty("user.home")
                //Helper.getPropValues("dl4j_home.data")
        ).toString();*/

        File parentDir = new ClassPathResource("drawings").getFile();

        trainDir = Paths.get(System.getProperty("user.home"),"cdle-group-3/project/src/main/resources");
        testDir = Paths.get(System.getProperty("user.home"), "cdle-group-3/project/src/main/resources");

        //Files in directories under the parent dir that have "allowed extensions" split needs a random number generator for reproducibility when splitting the files into train and test
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rng);

        //The balanced path filter gives you fine tune control of the min/max cases to load for each class
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        if (trainPerc >= 100) {
            throw new IllegalArgumentException("Percentage of data set aside for training has to be less than 100%. Test percentage = 100 - training percentage, has to be greater than 0");
        }

        //Split the image files into train and test
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPerc, 100-trainPerc);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];
    }

    /*private static void loadData() throws IOException {
        dataDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();
        downloadLink = Helper.getPropValues("dataset.fruits.url");
        File parentDir = new File(Paths.get(dataDir, "fruits").toString());
        if (!parentDir.exists()) {
            downloadAndUnzip();
        }
    }

//    private static void downloadAndUnzip() throws IOException {
//        String dataPath = new File(dataDir).getAbsolutePath();
//        File zipFile = new File(dataPath, "fruits-detection.zip");
//
//        if (!zipFile.isFile()) {
//            log.info("Downloading the dataset from " + downloadLink + "...");
//            FileUtils.copyURLToFile(new URL(downloadLink), zipFile);
//        }
//        ArchiveUtils.unzipFileTo(zipFile.getAbsolutePath(), dataPath);
//    }

    private static void downloadAndUnzip() throws IOException {
        String dataPath = new File(dataDir).getAbsolutePath();
        File zipFile = new File(dataPath, "fruits-detection.zip");

        log.info("Downloading the dataset from "+downloadLink+ "...");
        FileUtils.copyURLToFile(new URL(downloadLink), zipFile);

        if(!Helper.getCheckSum(zipFile.getAbsolutePath())
                .equalsIgnoreCase(Helper.getPropValues("dataset.fruits.hash"))){
            log.info("Downloaded file is incomplete");
            System.exit(0);
        }

        log.info("Unzipping "+zipFile.getAbsolutePath());
        ArchiveUtils.unzipFileTo(zipFile.getAbsolutePath(), dataPath);
    }*/
}
