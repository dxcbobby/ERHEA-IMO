package Opponent;

import java.util.*;

import DL.DeepModel;
import enumerate.Action;

import struct.FrameData;
import struct.CharacterData;
import struct.GameData;
import util.Parameter;
import struct.AttackData;

public class DLOpponentModel {

	private DeepModel dlModel;

	private Vector cur_batch_feature;
	private Vector nx_batch_feature;
	private Vector batch_sel_act_label;
	private Vector batch_opp_sel_act_label;

	private float[] last_feature;
	static private int batch_size = 100;
	static private int batch_pool_size = 1000;

	LinkedList<Float> input_data;

	// config maximum of values
	float max_hp;
	float max_energy;
	float stage_height;
	float stage_width;
	Deque<AttackData> myAttack;
	Deque<AttackData> oppAttack;

	float[] my_CurAttack;
	float[] opp_CurAttack;

	public DLOpponentModel(GameData gd, FrameData fd, String myName, String oppName, String lossType) {
		this.input_data = new LinkedList<Float>();
		this.last_feature = new float[26];
		for (int i = 0; i < this.last_feature.length; i++) {
			this.last_feature[i] = 0;
		}
		int input_size = this.last_feature.length;

		dlModel = new DeepModel(input_size, myName, oppName, lossType);
		// batch features and labels
		cur_batch_feature = new Vector();
		nx_batch_feature = new Vector();
		batch_sel_act_label = new Vector();
		batch_opp_sel_act_label = new Vector();

		// sets
		max_hp = (float) (gd.getMaxHP(true));
		max_energy = (float) (gd.getMaxEnergy(true));
		stage_height = (float) (gd.getStageHeight());
		stage_width = (float) (gd.getStageWidth());

	}

	private float[] preprocess(GameData gd, FrameData fd, boolean player) {
		this.input_data.clear();

		// current state.
		CharacterData my = fd.getCharacter(player);
		CharacterData opp = fd.getCharacter(!player);

		// my part
		float myHp = (float) (my.getHp());
		float myEnergy = (float) (my.getEnergy());
		float myPosX = (float) (my.getCenterX());
		float myPosY = (float) (my.getCenterY());
		float mySpeedX = (float) (my.getSpeedX());
		float mySpeedY = (float) (my.getSpeedY());
		float myHits = (float) (my.getHitCount());
		float myRemainFrame = (float) (my.getRemainingFrame());
		int myState = my.getState().ordinal();

		// opp part
		float oppHp = (float) (opp.getHp());
		float oppEnergy = (float) (opp.getEnergy());
		float oppPosX = (float) (opp.getCenterX());
		float oppPosY = (float) (opp.getCenterY());
		float oppSpeedX = (float) (opp.getSpeedX());
		float oppSpeedY = (float) (opp.getSpeedY());
		float oppHits = (float) (opp.getHitCount());
		float oppRemainFrame = (float) (opp.getRemainingFrame());
		int oppState = opp.getState().ordinal();

		// dist x, y
		float distX = (float) (fd.getDistanceX());
		float distY = (float) (fd.getDistanceY());
		int stateLen = (my.getState().values().length);
		float maxHitCount = 10;

		if (player) {
			myAttack = fd.getProjectilesByP1();
			oppAttack = fd.getProjectilesByP2();
		} else {
			myAttack = fd.getProjectilesByP2();
			oppAttack = fd.getProjectilesByP1();
		}

		// my- 7
		myHp = myHp / this.max_hp;
		myEnergy = myEnergy / this.max_energy;
		myPosX = (myPosX - this.stage_width / 2) / (this.stage_width / 2);
		myPosY = (myPosY / this.stage_height);
		mySpeedX = mySpeedX / 20;
		mySpeedY = mySpeedY / 28;
		myRemainFrame = myRemainFrame / 70;

		// opp- 7
		oppHp = oppHp / this.max_hp;
		oppEnergy = oppEnergy / this.max_energy;
		oppPosX = (oppPosX - this.stage_width / 2) / (this.stage_width / 2);
		oppPosY = (oppPosY / this.stage_height);
		oppSpeedX = oppSpeedX / 20;
		oppSpeedY = oppSpeedY / 28;
		oppRemainFrame = oppRemainFrame / 70;

		// 2
		distX = (distX - this.stage_width / 2) / (this.stage_width / 2);
		distY = (distY / this.stage_height);

		// 2
		myHits = Math.min(myHits, maxHitCount) / maxHitCount;
		oppHits = Math.min(oppHits, maxHitCount) / maxHitCount;

		// append data
		// hp part
		this.input_data.add(myHp);
		this.input_data.add(oppHp);

		// my part
		this.input_data.add(myEnergy);
		this.input_data.add(myPosX);
		this.input_data.add(myPosY);
		this.input_data.add(mySpeedX);
		this.input_data.add(mySpeedY);
		this.input_data.add(myRemainFrame);
		this.input_data.add(myHits);
		for (int i = 0; i < stateLen; i++) {
			if (myState == i) {
				this.input_data.add(1.f);
			} else {
				this.input_data.add(0.f);
			}
		}
		// opp part
		this.input_data.add(oppEnergy);
		this.input_data.add(oppPosX);
		this.input_data.add(oppPosY);
		this.input_data.add(oppSpeedX);
		this.input_data.add(oppSpeedY);
		this.input_data.add(oppRemainFrame);
		this.input_data.add(oppHits);
		for (int i = 0; i < stateLen; i++) {
			if (oppState == i) {
				this.input_data.add(1.f);
			} else {
				this.input_data.add(0.f);
			}
		}

		// dist
		this.input_data.add(distX);
		this.input_data.add(distY);

		int len_inputs = this.input_data.size();
		float[] outs = new float[len_inputs];
		for (int i = 0; i < len_inputs; i++) {
			outs[i] = this.input_data.get(i);
		}

		return outs;
	}

	public Action predict(GameData gd, FrameData fd, boolean player) {
//		float[] inputs = preprocess(gd, fd, player);
		float[] inputs = getInputData(fd, player);
		int actionIdx = dlModel.forward(inputs);
//		Action act = Action.values()[actionIdx];
		Action act = Parameter.NormalizedAction[actionIdx];
		return act;
	}

	public Action predict(GameData gd, FrameData fd, boolean player, LinkedList<Action> validAction) {
//		float[] inputs = preprocess(gd, fd, player);
		float[] inputs = getInputData(fd, player);
		int actionIdx = dlModel.forward(inputs, validAction);
//		Action act = Action.values()[actionIdx];
		Action act = Parameter.NormalizedAction[actionIdx];
		return act;
	}

//	public void record(GameData gd, FrameData cur_fd, FrameData nx_fd, boolean player, int sel_act, int opp_sel_act){
//		float[] cur_inputs = preprocess(gd, cur_fd, player);
//		float[] nx_inputs = preprocess(gd, nx_fd, player);
//		
//		for (int i=0; i<cur_inputs.length; i++){
//			if (cur_inputs[i] != this.last_feature[i]){
//				cur_batch_feature.addElement(cur_inputs);
//				nx_batch_feature.addElement(nx_inputs);
//				batch_sel_act_label.addElement((float)(sel_act));
//				batch_opp_sel_act_label.addElement((float)(opp_sel_act));
//				
//				this.last_feature = cur_inputs;
//				break;
//			}
//		}
//		
//		
//	}

//	public void train_batch(float win_signal) {
//		int size = cur_batch_feature.size();
//		if (size > 0) {
//			float[][] cur_input_features = new float[size][];
//			float[][] nx_input_features = new float[size][];
//			float[] sel_act_labels = new float[size];
//			float[] opp_sel_act_labels = new float[size];
//			for (int i = 0; i < size; i++) {
//				cur_input_features[i] = (float[]) cur_batch_feature.get(i);
//				nx_input_features[i] = (float[]) nx_batch_feature.get(i);
//				sel_act_labels[i] = (float) batch_sel_act_label.get(i);
//				opp_sel_act_labels[i] = (float) batch_opp_sel_act_label.get(i);
//
//			}
//			dlModel.train(cur_input_features, nx_input_features, sel_act_labels, opp_sel_act_labels, win_signal);
//			cur_batch_feature.clear();
//			nx_batch_feature.clear();
//			batch_sel_act_label.clear();
//			batch_opp_sel_act_label.clear();
//
//		}
//
//		// clear last feature.
//		for (int i = 0; i < this.last_feature.length; i++) {
//			this.last_feature[i] = 0;
//		}
//		System.out.println("Finish Training Batch!");
//	}

	public void saveModel() {
		dlModel.save();
	}

	public void loadModel() {
		dlModel.load();
	}

	public static float[] getInputData(FrameData frameData, boolean player) {
		// TODO Auto-generated method stub
		ArrayList<Double> data_list = new ArrayList<>();
		float inputData[] = new float[Parameter.inputNum];
		CharacterData my = frameData.getCharacter(player);
		CharacterData opp = frameData.getCharacter(!player);

		// time info
		double game_frame_num = (double) frameData.getFramesNumber();
		double myPlayer = -1;
		if (player) {
			myPlayer = 0;
		} else {
			myPlayer = 1;
		}

		double disX = (double) frameData.getDistanceX();
		double disY = (double) frameData.getDistanceY();

		// my info
		double myState = (double) my.getState().ordinal();
		double myHp = (double) my.getHp();

		double myEnergy = (double) my.getEnergy();
		double myX = ((double) my.getLeft() + (double) my.getRight()) / 2;
		double myY = ((double) my.getBottom() + (double) my.getTop()) / 2;
		double mySpeedX = (double) my.getSpeedX();
		double mySpeedY = (double) my.getSpeedY();
		double myAction = Double.valueOf(my.getAction().ordinal());
		double myRemainingFrame = (double) my.getRemainingFrame() / 70;
		double myFront = -1;
		if (my.isFront()) {
			myFront = 1;
		} else {
			myFront = 0;
		}
		;
		double myControl = -1;

		if (my.isControl()) {
			myControl = 1;
		} else {
			myControl = 0;
		}
		;
		double myHit = -1;
		if (my.isHitConfirm()) {
			myHit = 1;
		} else {
			myHit = 0;
		}
		;

		// opp info
		double oppState = (double) opp.getState().ordinal();
		double oppHp = (double) opp.getHp();
		double oppEnergy = (double) opp.getEnergy();
		double oppX = ((double) opp.getLeft() + (double) opp.getRight()) / 2;
		double oppY = ((double) opp.getBottom() + (double) opp.getTop()) / 2;
		double oppSpeedX = (double) opp.getSpeedX();
		double oppSpeedY = (double) opp.getSpeedY();
		double oppAction = Double.valueOf(opp.getAction().ordinal());
		double oppRemainingFrame = (double) opp.getRemainingFrame();
		double oppFront = -1;
		if (opp.isFront()) {
			oppFront = 1;
		} else {
			oppFront = 0;
		}
		;
		double oppControl = -1;
		if (opp.isControl()) {
			oppControl = 1;
		} else {
			oppControl = 0;
		}

		double oppHit = -1;
		if (opp.isHitConfirm()) {
			oppHit = 1;
		} else {
			oppHit = 0;
		}

//		// projectile
//		List<AttackData> myAttack = new ArrayList<>(
//				player ? frameData.getProjectilesByP1() : frameData.getProjectilesByP2());
//		List<AttackData> oppAttack = new ArrayList<>(
//				player ? frameData.getProjectilesByP2() : frameData.getProjectilesByP1());

		// data record to list
		inputData[0] = (float) myPlayer;// 0
		inputData[1] = (float) disX / 960;// 1
		inputData[2] = (float) disY / 640;// 2

		inputData[3] = (float) myAction / 55;// 3
		inputData[4] = (float) oppAction / 55;// 4

		// 5-15
		inputData[5] = (float) myState / 3;// 5
		inputData[6] = (float) myHp / 400;// 6
		inputData[7] = (float) myEnergy / 300;// 7
		inputData[8] = (float) myX / 960;// 8
		inputData[9] = (float) myY / 640;// 9
		inputData[10] = (float) mySpeedX / 15;// 10
		inputData[11] = (float) mySpeedY / 28;// 11
		inputData[12] = (float) myRemainingFrame / 70;// 12
		inputData[13] = (float) myFront;// 13
		inputData[14] = (float) myControl;// 14
		inputData[15] = (float) myHit;// 15

		// 16-26
		inputData[16] = (float) oppState / 3;// 16
		inputData[17] = (float) oppHp / 400;// 17
		inputData[18] = (float) oppEnergy / 300;// 18
		inputData[19] = (float) oppX / 960;// 19
		inputData[20] = (float) oppY / 640;// 20
		inputData[21] = (float) oppSpeedX / 15;// 21
		inputData[22] = (float) oppSpeedY / 28;// 22
		inputData[23] = (float) oppRemainingFrame / 70;// 23
		inputData[24] = (float) oppFront;// 24
		inputData[25] = (float) oppControl;// 25
		inputData[26] = (float) oppHit;// 26

//		// 29 - 34
//		for (int i = 0; i < 2; i++) {
//			if (myAttack.size() > i) {
//				AttackData tmp = myAttack.get(i);
//				data_list.add((double) tmp.getHitDamage() / 200.0);
//				data_list.add(
//						(((double) tmp.getCurrentHitArea().getLeft() + (double) tmp.getCurrentHitArea().getRight()) / 2)
//								/ 960.0);
//				data_list.add(
//						(((double) tmp.getCurrentHitArea().getTop() + (double) tmp.getCurrentHitArea().getBottom()) / 2)
//								/ 640.0);
//			} else {
//				data_list.add(0.0);
//				data_list.add(0.0);
//				data_list.add(0.0);
//			}
//		}
//		for (int i = 0; i < 2; i++) {
//			if (oppAttack.size() > i) {
//				AttackData tmp = oppAttack.get(i);
//				data_list.add((double) tmp.getHitDamage() / 200.0);
//				data_list.add(
//						(((double) tmp.getCurrentHitArea().getLeft() + (double) tmp.getCurrentHitArea().getRight()) / 2)
//								/ 960.0);
//				data_list.add(
//						(((double) tmp.getCurrentHitArea().getTop() + (double) tmp.getCurrentHitArea().getBottom()) / 2)
//								/ 640.0);
//			} else {
//				data_list.add(0.0);
//				data_list.add(0.0);
//				data_list.add(0.0);
//			}
//		}
		return inputData;
	}

}