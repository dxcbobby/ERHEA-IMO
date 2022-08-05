package util;

import enumerate.Action;

public class Parameter {
	// System Setting
	public static long budgetTime = (long) (16.5 * 1000000 - 30);
	public static double judePoint = 0.75; // If Imitator use half, then not use RHEA.

	// MLPModel and file Path
	public static String aiName = "BlackMamba";
	public static String trainDataNum = "800";
	public static String modelVersion = "85";
//	public static String lossType = "MCXENT"; // multi-class cross entropy loss
	public static int inputNum = 27;
	public static int outputNum = 42;

	// OM Model
	public String lossType = "pi"; // pi, sl
	public static int epoch = 10;

	public static String modelPathName = "./data/aiData/IOM/IOM0.zip";
	public static String timePath = "./data/aiData/ERHEASI/" + trainDataNum + "-" + aiName + "-" + modelVersion
			+ "-" + "Time.csv";
	public static String actionPath = "./data/aiData/ERHEASI/" + trainDataNum + "-" + aiName + "-" + modelVersion
			+ "-" + "Action.csv";
//	public static String modelPathName = "C:/Bobby/programming/Dataset/Model/" + trainDataNum + "-" + aiName + "-"
//			+ modelVersion + "-" + "ImitatorModel.zip";
//	public static String timePath = "C:/Bobby/programming/Dataset/Record/" + trainDataNum + "-" + aiName + "-"
//			+ modelVersion + "-" + "Time.csv";
//	public static String actionPath = "C:/Bobby/programming/Dataset/Record/" + trainDataNum + "-" + aiName + "-"
//			+ modelVersion + "-" + "Action.csv";

	// RHEA
	public static int populationSize = 5; // 4,5
	public static int individualSize = 2; // 3,2
	public static String crossoverType = "Uniform";
	public static boolean shiftLeft = true;
	public static boolean omModel = false;
	public static double mutationRate = 0.5;

	// Action Setting
	public static Action[] NormalizedAction = new Action[] { Action.NEUTRAL, Action.CROUCH, Action.AIR_A, Action.AIR_B,
			Action.AIR_D_DB_BA, Action.AIR_D_DB_BB, Action.AIR_D_DF_FA, Action.AIR_D_DF_FB, Action.AIR_DA,
			Action.AIR_DB, Action.AIR_F_D_DFA, Action.AIR_F_D_DFB, Action.AIR_FA, Action.AIR_FB, Action.AIR_GUARD,
			Action.AIR_UA, Action.AIR_UB, Action.BACK_JUMP, Action.BACK_STEP, Action.CROUCH_A, Action.CROUCH_B,
			Action.CROUCH_FA, Action.CROUCH_FB, Action.CROUCH_GUARD, Action.DASH, Action.FOR_JUMP, Action.FORWARD_WALK,
			Action.JUMP, Action.STAND_A, Action.STAND_B, Action.STAND_D_DB_BA, Action.STAND_D_DB_BB,
			Action.STAND_D_DF_FA, Action.STAND_D_DF_FB, Action.STAND_D_DF_FC, Action.STAND_F_D_DFA,
			Action.STAND_F_D_DFB, Action.STAND_FA, Action.STAND_FB, Action.STAND_GUARD, Action.THROW_A,
			Action.THROW_B };

}
