import torch
import torch.nn as nn

class FullAdaptedWaterNet(nn.Module):
    def __init__(self, input_channels=3):
        super(FullAdaptedWaterNet, self).__init__()

        # Bloque de procesamiento de imagen RAW
        self.conv_raw1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn_raw1 = nn.BatchNorm2d(64)
        self.relu_raw1 = nn.ReLU()
        self.conv_raw2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_raw2 = nn.BatchNorm2d(128)
        self.relu_raw2 = nn.ReLU()

        # Bloque de procesamiento de imagen MSRCR
        self.conv_msrcr1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn_msrcr1 = nn.BatchNorm2d(64)
        self.relu_msrcr1 = nn.ReLU()
        self.conv_msrcr2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_msrcr2 = nn.BatchNorm2d(128)
        self.relu_msrcr2 = nn.ReLU()

        # Bloque de procesamiento de balance de blancos (WB)
        self.conv_wb1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn_wb1 = nn.BatchNorm2d(64)
        self.relu_wb1 = nn.ReLU()
        self.conv_wb2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_wb2 = nn.BatchNorm2d(128)
        self.relu_wb2 = nn.ReLU()

        # Bloque de procesamiento de equalización de histograma (HE)
        self.conv_he1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn_he1 = nn.BatchNorm2d(64)
        self.relu_he1 = nn.ReLU()
        self.conv_he2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_he2 = nn.BatchNorm2d(128)
        self.relu_he2 = nn.ReLU()

        # Bloque de procesamiento de compensación de gamma (GC)
        self.conv_gc1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn_gc1 = nn.BatchNorm2d(64)
        self.relu_gc1 = nn.ReLU()
        self.conv_gc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_gc2 = nn.BatchNorm2d(128)
        self.relu_gc2 = nn.ReLU()

        # Fusión de características con mayor ponderación para MSRCR
        self.conv_fusion1 = nn.Conv2d(128 * 5, 256, kernel_size=3, padding=1)
        self.bn_fusion1 = nn.BatchNorm2d(256)
        self.relu_fusion1 = nn.ReLU()
        self.dropout_fusion1 = nn.Dropout(p=0.5)
        self.conv_fusion2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn_fusion2 = nn.BatchNorm2d(128)
        self.relu_fusion2 = nn.ReLU()
        self.dropout_fusion2 = nn.Dropout(p=0.5)
        self.conv_fusion3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn_fusion3 = nn.BatchNorm2d(64)
        self.relu_fusion3 = nn.ReLU()
        self.conv_fusion4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, raw_image, he_image, gc_image, wb_image, msrcr_image):
        # Procesamiento de imagen RAW
        raw = self.relu_raw1(self.bn_raw1(self.conv_raw1(raw_image)))
        raw = self.relu_raw2(self.bn_raw2(self.conv_raw2(raw)))

        # Procesamiento de imagen MSRCR
        msrcr = self.relu_msrcr1(self.bn_msrcr1(self.conv_msrcr1(msrcr_image)))
        msrcr = self.relu_msrcr2(self.bn_msrcr2(self.conv_msrcr2(msrcr)))

        # Procesamiento WB
        wb = self.relu_wb1(self.bn_wb1(self.conv_wb1(wb_image)))
        wb = self.relu_wb2(self.bn_wb2(self.conv_wb2(wb)))

        # Procesamiento HE
        he = self.relu_he1(self.bn_he1(self.conv_he1(he_image)))
        he = self.relu_he2(self.bn_he2(self.conv_he2(he)))

        # Procesamiento GC
        gc = self.relu_gc1(self.bn_gc1(self.conv_gc1(gc_image)))
        gc = self.relu_gc2(self.bn_gc2(self.conv_gc2(gc)))

        # Concatenación de características, ponderando MSRCR
        fused = torch.cat((raw, msrcr * 1.5, wb, he, gc), dim=1)  # MSRCR ponderado más alto
        fused = self.relu_fusion1(self.bn_fusion1(self.conv_fusion1(fused)))
        fused = self.dropout_fusion1(fused)
        fused = self.relu_fusion2(self.bn_fusion2(self.conv_fusion2(fused)))
        fused = self.dropout_fusion2(fused)
        fused = self.relu_fusion3(self.bn_fusion3(self.conv_fusion3(fused)))
        output = self.conv_fusion4(fused)

        return output
