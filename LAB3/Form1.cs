using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.OCR;                  // модуль оптического распознавания символов
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Windows.Forms;

namespace LAB3
{
    public partial class Form1 : Form
    {
        // === Захват видео ===
        private VideoCapture capture;

        // === OCR ===
        private readonly Tesseract _ocr;
        private readonly string tessdataPath = @"C:\Users\GOYdenish\source\repos\tessdata"; // путь к данным обучения (github.com/tesseract-ocr/tessdata)

        // === Обнаружение лиц ===
        private readonly string cascadePath = @"C:\Users\GOYdenish\source\repos\LAB3\haarcascade_frontalface_default.xml";

        // === Данные ===
        private Mat sourceImage;                                  // загруженное изображение
        private readonly List<Rectangle> textRegions = new List<Rectangle>();
        private Mat maskMat;                                      // маска (4 канала, с прозрачностью)

        // === Режимы обработки видео ===
        private bool detectFaces = false;
        private bool applyMask = false;

        // === Элементы интерфейса ===
        private PictureBox pbOriginal;   // "оригинальное" — слева
        private PictureBox pbResult;     // "результат обработки" — справа
        private Button btnLoadImage;
        private Button btnWebCamera;
        private Button btnLoadVideo;
        private Button btnStart;
        private Button btnStop;
        private Button btnDetectText;
        private Button btnRecognize;
        private Button btnOcrFrame;      // OCR текущего кадра видео
        private Button btnLoadMask;
        private ComboBox cbRegions;
        private TextBox tbText;
        private Timer timer1;

        public Form1()
        {
            Text = "Lab3: поиск текста и лиц";
            ClientSize = new Size(1200, 660);

            pbOriginal = new PictureBox
            {
                Location = new Point(10, 10),
                Size = new Size(580, 480),
                SizeMode = PictureBoxSizeMode.Zoom,
                BorderStyle = BorderStyle.FixedSingle
            };
            pbResult = new PictureBox
            {
                Location = new Point(605, 10),
                Size = new Size(580, 480),
                SizeMode = PictureBoxSizeMode.Zoom,
                BorderStyle = BorderStyle.FixedSingle
            };

            btnLoadImage = new Button { Text = "Load Image", Location = new Point(10, 500), Width = 110 };
            btnLoadImage.Click += BtnLoadImage_Click;

            btnWebCamera = new Button { Text = "Web Camera", Location = new Point(130, 500), Width = 110 };
            btnWebCamera.Click += BtnWebCamera_Click;

            btnLoadVideo = new Button { Text = "Load Video", Location = new Point(250, 500), Width = 110 };
            btnLoadVideo.Click += BtnLoadVideo_Click;

            btnStart = new Button { Text = "Start", Location = new Point(370, 500), Width = 90 };
            btnStart.Click += BtnStart_Click;

            btnStop = new Button { Text = "Stop", Location = new Point(470, 500), Width = 90 };
            btnStop.Click += BtnStop_Click;

            btnDetectText = new Button { Text = "Find text", Location = new Point(605, 500), Width = 110 };
            btnDetectText.Click += BtnDetectText_Click;

            btnRecognize = new Button { Text = "Recognize", Location = new Point(725, 500), Width = 110 };
            btnRecognize.Click += BtnRecognize_Click;

            btnOcrFrame = new Button { Text = "OCR frame", Location = new Point(845, 500), Width = 110 };
            btnOcrFrame.Click += BtnOcrFrame_Click;

            btnLoadMask = new Button { Text = "Load mask", Location = new Point(965, 500), Width = 110 };
            btnLoadMask.Click += BtnLoadMask_Click;

            cbRegions = new ComboBox
            {
                Location = new Point(605, 540),
                Width = 130,
                DropDownStyle = ComboBoxStyle.DropDownList
            };
            cbRegions.SelectedIndexChanged += CbRegions_SelectedIndexChanged;

            tbText = new TextBox
            {
                Location = new Point(605, 575),
                Size = new Size(580, 60),
                Multiline = true,
                ReadOnly = true,
                ScrollBars = ScrollBars.Vertical
            };

            Controls.AddRange(new Control[]
            {
                pbOriginal, pbResult,
                btnLoadImage, btnWebCamera, btnLoadVideo, btnStart, btnStop,
                btnDetectText, btnRecognize, btnOcrFrame, btnLoadMask,
                cbRegions, tbText
            });

            // таймер для захвата кадров (интервал из задания — 35 мс)
            timer1 = new Timer { Interval = 35, Enabled = false };
            timer1.Tick += Timer1_Tick;

            // инициализация OCR
            _ocr = new Tesseract(tessdataPath, "eng", OcrEngineMode.TesseractLstmCombined);
        }

        // =====================================================================
        // 1. Выделение участков изображения, потенциально содержащих текст
        // =====================================================================
        private void BtnDetectText_Click(object sender, EventArgs e)
        {
            if (sourceImage == null || sourceImage.IsEmpty) return;

            // монохромное изображение с чёрным фоном (пороговое преобразование)
            using (Mat gray = new Mat())
            using (Mat thresh = new Mat())
            {
                CvInvoke.CvtColor(sourceImage, gray, ColorConversion.Bgr2Gray);
                CvInvoke.Threshold(gray, thresh, 128, 255, ThresholdType.BinaryInv);
                // "расширение" отдельных букв для объединения в области
                CvInvoke.Dilate(thresh, thresh, null, new Point(-1, -1), 5, BorderType.Constant, new MCvScalar());

                // выделение областей интереса: контуры и их ограничивающие прямоугольники
                using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
                {
                    CvInvoke.FindContours(thresh, contours, null,
                        RetrType.List, ChainApproxMethod.ChainApproxSimple);

                    using (Mat output = sourceImage.Clone())
                    {
                        textRegions.Clear();
                        cbRegions.Items.Clear();

                        for (int i = 0; i < contours.Size; i++)
                        {
                            using (VectorOfPoint contour = contours[i])
                            {
                                if (CvInvoke.ContourArea(contour, false) > 50) // игнорирование маленьких контуров
                                {
                                    Rectangle rect = CvInvoke.BoundingRectangle(contour);
                                    textRegions.Add(rect);
                                    CvInvoke.Rectangle(output, rect, new MCvScalar(255, 0, 0), 1); // синяя рамка
                                    cbRegions.Items.Add("Region " + i);
                                }
                            }
                        }

                        ShowOn(pbOriginal, sourceImage);
                        ShowOn(pbResult, output);
                    }
                }
            }

            if (cbRegions.Items.Count > 0) cbRegions.SelectedIndex = 0;
        }

        // =====================================================================
        // 2. Выбор и отображение участка изображения
        // =====================================================================
        private void CbRegions_SelectedIndexChanged(object sender, EventArgs e)
        {
            int index = cbRegions.SelectedIndex;
            if (sourceImage == null || index < 0 || index >= textRegions.Count) return;

            // копия участка изображения (Mat поверх ROI)
            using (Mat roi = new Mat(sourceImage, textRegions[index]))
            using (Mat roiImg = roi.Clone())
                ShowOn(pbResult, roiImg);
        }

        // =====================================================================
        // 3. Получение текста, содержащегося на выбранном участке
        // =====================================================================
        private void BtnRecognize_Click(object sender, EventArgs e)
        {
            int index = cbRegions.SelectedIndex;
            if (sourceImage == null || index < 0 || index >= textRegions.Count) return;

            using (Mat roi = new Mat(sourceImage, textRegions[index]))
            using (Mat roiImg = roi.Clone())
            {
                _ocr.SetImage(roiImg);   // фрагмент изображения, содержащий текст
                _ocr.Recognize();        // распознавание текста
                tbText.Text = _ocr.GetUTF8Text().Trim(); // результат строкой
            }
        }

        // =====================================================================
        // 4. Распознавание текста с текущего кадра видео
        // =====================================================================
        private void BtnOcrFrame_Click(object sender, EventArgs e)
        {
            if (capture == null) return;

            using (Mat frame = capture.QueryFrame())
            {
                if (frame == null || frame.IsEmpty) return;

                _ocr.SetImage(frame);
                _ocr.Recognize();
                tbText.Text = _ocr.GetUTF8Text().Trim();
            }
        }

        // =====================================================================
        // Захват видео с веб-камеры
        // =====================================================================
        private void BtnWebCamera_Click(object sender, EventArgs e)
        {
            if (capture != null) capture.Dispose();
            capture = new VideoCapture();       // инициализация веб-камеры
            timer1.Enabled = true;              // начало обработки видеопотока
        }

        // =====================================================================
        // Захват видео из файла
        // =====================================================================
        private void BtnLoadVideo_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog f = new OpenFileDialog())
            {
                f.Filter = "Video|*.mp4;*.avi;*.wmv;*.mkv";
                if (f.ShowDialog() == DialogResult.OK)
                {
                    if (capture != null) capture.Dispose();
                    capture = new VideoCapture(f.FileName);

                    // информация о видео
                    double frameCount = capture.Get(CapProp.FrameCount);  // количество кадров
                    double frameRate = capture.Get(CapProp.Fps);         // кадров в секунду
                    double frameHeight = capture.Get(CapProp.FrameHeight); // высота кадров
                    double frameWidth = capture.Get(CapProp.FrameWidth);  // ширина кадров
                    Text = $"Lab3 — {(int)frameCount} кадров, {frameRate:F1} к/с, {(int)frameWidth}x{(int)frameHeight}";
                }
            }
        }

        // Захват кадра по событию Tick таймера (и для камеры, и для файла)
        private void Timer1_Tick(object sender, EventArgs e)
        {
            if (capture == null) return;

            using (Mat frame = capture.QueryFrame())
            {
                if (frame == null || frame.IsEmpty)
                {
                    BtnStop_Click(null, null);      // видео закончилось
                    return;
                }
                ShowFrame(frame);
            }
        }

        private void BtnStart_Click(object sender, EventArgs e)
        {
            if (capture != null)
                timer1.Enabled = true;              // активация таймера из кода
        }

        private void BtnStop_Click(object sender, EventArgs e)
        {
            timer1.Enabled = false;
            if (capture != null)
            {
                capture.Stop();                     // остановка обработки видеопотока
                capture.Dispose();
                capture = null;
            }
        }

        // =====================================================================
        // Обработка кадра: обнаружение и обозначение лиц, наложение масок
        // =====================================================================
        private void ShowFrame(Mat frame)
        {
            using (Mat res = frame.Clone())
            {
                if (detectFaces)
                {
                    using (CascadeClassifier face = new CascadeClassifier(cascadePath))
                    using (Mat ugray = new Mat())
                    {
                        CvInvoke.CvtColor(frame, ugray, ColorConversion.Bgr2Gray);
                        // параметры DetectMultiScale:
                        // 1) чёрно-белый вариант целевого изображения
                        // 2) коэффициент масштабирования (> 1.0; ближе к 1.0 — чувствительнее и медленнее)
                        // 3) минимальное число ближайших соседей (больше — меньше ложных срабатываний)
                        Rectangle[] facesDetected = face.DetectMultiScale(ugray, 1.1, 10, new Size(20, 20), Size.Empty);

                        foreach (Rectangle rectF in facesDetected)
                        {
                            // обозначение найденных лиц (жёлтая рамка)
                            CvInvoke.Rectangle(res, rectF, new MCvScalar(0, 255, 255), 2);

                            if (applyMask && maskMat != null && maskMat.IsEmpty == false)
                                ApplyMask(res, rectF);       // наложение маски на лицо
                        }
                    }
                }

                ShowOn(pbOriginal, frame);   // оригинал слева
                ShowOn(pbResult, res);       // результат обработки справа
            }
        }

        // Наложение маски на область лица: маска растягивается до размера лица,
        // прозрачные пиксели (альфа = 0) не перекрывают кадр
        private void ApplyMask(Mat res, Rectangle rectF)
        {
            // обрезаем прямоугольник, чтобы он не выходил за границы кадра
            rectF.Intersect(new Rectangle(Point.Empty, res.Size));
            if (rectF.Width <= 0 || rectF.Height <= 0) return;

            using (Mat patch = new Mat(res, rectF))          // область кадра с лицом
            using (Mat small = new Mat())
            {
                CvInvoke.Resize(maskMat, small, patch.Size); // маска под размер лица

                if (small.NumberOfChannels == 4)
                {
                    Mat[] channels = small.Split();
                    try
                    {
                        using (Mat alpha = channels[3])                       // альфа-канал как маска
                        using (Mat maskBin = new Mat())
                        using (Mat smallBgr = new Mat())
                        {
                            CvInvoke.Threshold(alpha, maskBin, 128, 255, ThresholdType.Binary);
                            CvInvoke.CvtColor(small, smallBgr, ColorConversion.Bgra2Bgr);
                            // копирование small на patch с учётом маски
                            smallBgr.CopyTo(patch, maskBin);
                        }
                    }
                    finally
                    {
                        foreach (Mat ch in channels) ch.Dispose();
                    }
                }
                else
                {
                    small.CopyTo(patch);          // маска без прозрачности — целиком
                }
            }
        }

        // =====================================================================
        // Загрузка изображения (для работы с текстом)
        // =====================================================================
        private void BtnLoadImage_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog f = new OpenFileDialog())
            {
                if (f.ShowDialog() == DialogResult.OK)
                {
                    // (ImreadModes)1 == режим цветного чтения; числовое значение стабильно,
                    // что страхует от переименований членов перечисления в разных версиях
                    Mat img = CvInvoke.Imread(f.FileName, (ImreadModes)1);
                    if (img.IsEmpty) return;

                    if (sourceImage != null) sourceImage.Dispose();
                    sourceImage = img;

                    textRegions.Clear();
                    cbRegions.Items.Clear();
                    tbText.Text = "";

                    ShowOn(pbOriginal, sourceImage);
                    pbResult.Image = null;
                }
            }
        }

        // =====================================================================
        // Загрузка маски (изображение с поддержкой прозрачности)
        // =====================================================================
        private void BtnLoadMask_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog f = new OpenFileDialog())
            {
                f.Filter = "Image|*.png;*.jpg;*.bmp";
                if (f.ShowDialog() == DialogResult.OK)
                {
                    // (ImreadModes)(-1) == "Unchanged": читаем как есть, с альфа-каналом
                    Mat m = CvInvoke.Imread(f.FileName, (ImreadModes)(-1));
                    if (m.IsEmpty) return;

                    if (maskMat != null) maskMat.Dispose();
                    maskMat = m;
                    detectFaces = true;   // после загрузки маски включаем обработку видео
                    applyMask = true;
                }
            }
        }

        // =====================================================================
        // Отображение Mat в PictureBox — собственный конвертер,
        // не требует пакета Emgu.CV.Bitmap
        // =====================================================================
        private static Bitmap MatToBitmap(Mat mat)
        {
            using (VectorOfByte buf = new VectorOfByte())
            {
                CvInvoke.Imencode(".png", mat, buf);   // PNG корректно кодирует и 1, и 3, и 4 канала
                using (MemoryStream ms = new MemoryStream(buf.ToArray()))
                using (Bitmap tmp = (Bitmap)Image.FromStream(ms))
                {
                    return new Bitmap(tmp);            // копия, независимая от потока
                }
            }
        }

        private void ShowOn(PictureBox pb, Mat m)
        {
            var old = pb.Image;
            pb.Image = MatToBitmap(m);
            old?.Dispose();
        }
    }
}
