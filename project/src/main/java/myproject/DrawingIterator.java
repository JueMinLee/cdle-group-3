package myproject;

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
import org.slf4j.Logger;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

public class DrawingIterator {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(DrawingIterator.class);
    private static final int seed = 123;
    private static Random rng = new Random(seed);
    private static String dataDir;
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
    private static int batchSize;

    private static RecordReaderDataSetIterator makeIterator(InputSplit split, boolean training,  int batchSize) throws Exception {

        ObjectDetectionRecordReader recordReader = new ObjectDetectionRecordReader(yoloheight, yolowidth, nChannels,
                gridHeight, gridWidth, new VocLabelProvider(dataDir));
        if (training && transform != null) {
            recordReader.initialize(split, transform);
        } else {
            recordReader.initialize(split);
        }
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, 1, true);
        iter.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        return iter;
    }

    public static RecordReaderDataSetIterator trainIterator(int batchSize) throws Exception {
        return makeIterator(trainData, true, batchSize);
    }

    public static RecordReaderDataSetIterator testIterator(int batchSize) throws Exception {
        return makeIterator(testData, false, batchSize);
    }

    public static void setup(int batchSizeArg, int trainPerc, ImageTransform imageTransform) throws IOException {
        transform = imageTransform;
        setup(batchSizeArg,trainPerc);
    }

    public static void setup(int batchSizeArg, int trainPerc) throws IOException {
        //For manually addd train and test folder
        /*trainDir = Paths.get(System.getProperty("user.home"), "cdle-group-3/project/src/main/resources/drawings", "train");
        testDir = Paths.get(System.getProperty("user.home"), "cdle-group-3/project/src/main/resources/drawings", "test");


        trainData = new FileSplit(new File(trainDir.toString()), NativeImageLoader.ALLOWED_FORMATS, rng);
        testData = new FileSplit(new File(testDir.toString()), NativeImageLoader.ALLOWED_FORMATS, rng);*/

        dataDir = new ClassPathResource("drawings").getFile().getPath();
        File parentDir = new File(Paths.get(dataDir).toString());

        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rng);

        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        batchSize = batchSizeArg;

        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPerc, 100-trainPerc);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];
    }

}


