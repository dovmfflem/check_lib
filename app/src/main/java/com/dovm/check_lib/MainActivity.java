package com.dovm.check_lib;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import com.lipiji.mllib.dataset.CharText;
import com.lipiji.mllib.rnn.lstm.LSTMLM;
import com.lipiji.mllib.layers.MatIniter;
import com.lipiji.mllib.layers.MatIniter.Type;

import java.io.File;


public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        int hiddenSize = 100;
        double lr = 1;


        CharText ct = new CharText(getFilesDir());
        LSTMLM lstm = new LSTMLM(ct.getCharIndex().size(), hiddenSize, new MatIniter(Type.Uniform, 0.1, 0, 0));
        lstm.train(ct, lr);

    }
}
